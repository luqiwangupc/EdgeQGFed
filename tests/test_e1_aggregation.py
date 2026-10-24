"""CPU-only E1 checks: synthetic tensors, no benchmark/weights/training/tracker imports."""

import ast
import copy
import json
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
from omegaconf import OmegaConf

from models.ema import EMA
from models.graph_aggregator import HierarchicalGraphAggregator
from tools.prepare_e1_configs import CORE_VARIANTS, VARIANTS, build_variant, main as prepare_configs
from utils.e1_logging import E1RunLogger
from utils.e1_random import e1_random_stream
from utils.communication import strict_budget_enabled


ROOT = Path(__file__).resolve().parents[1]


def config(mode='mpq_graph', teacher='sample_quality'):
    result = OmegaConf.create({'models': {'graph': {
        'use': True, 'temperature': 0.5, 'prototype_weight': 0.35,
        'reliability_weight': 0.35, 'label_ratio_weight': 0.2,
        'confidence_weight': 0.1, 'self_bias': 0.3, 'min_attention': 1e-4,
    }}})
    if mode is not None:
        result.models.graph.mode = mode
    if teacher is not None:
        result.models.graph.teacher_weight_mode = teacher
    return result


def summaries():
    return [{
        'name': name,
        'parameters': [torch.tensor([value], requires_grad=True), torch.tensor(count, dtype=torch.long)],
        'prototype': torch.tensor(prototype), 'prototype_counts': torch.ones(2),
        'pseudo_quality': rho, 'labeled_ratio': ratio, 'mean_confidence': confidence,
        'batch_size': n,
    } for name, value, count, prototype, rho, ratio, confidence, n in [
        ('Edge0', 0.0, 2, [[1., 0.], [0., 1.]], 1.0, .1, .9, 1),
        ('Edge1', 10.0, 7, [[0., 1.], [1., 0.]], 0.0, .8, .4, 3),
    ]]


class AggregationTests(unittest.TestCase):
    def test_sample_and_quality_direct_weights(self):
        for mode, expected, value in [
            ('sample_average', [.25, .75], 7.5),
            ('quality_average', [.4, .6], 6.0),
        ]:
            with self.subTest(mode=mode):
                source = summaries()
                result = HierarchicalGraphAggregator(config(mode, 'direct')).aggregate(source)
                torch.testing.assert_close(result['source_weights'], torch.tensor(expected))
                torch.testing.assert_close(result['attention_matrix'], torch.tensor([expected, expected]))
                self.assertAlmostEqual(result['global_parameters'][0].item(), value)
                for params in result['personalized_parameters'].values():
                    self.assertAlmostEqual(params[0].item(), value)
                    self.assertEqual(params[1].item(), 2)
                # Independent result tensors; aggregation never mutates input or keeps autograd.
                result['personalized_parameters']['Edge0'][0].add_(100)
                self.assertAlmostEqual(result['personalized_parameters']['Edge1'][0].item(), value)
                self.assertEqual(source[0]['parameters'][0].item(), 0.0)
                self.assertFalse(result['global_parameters'][0].requires_grad)

    def test_graph_direction_q_and_independent_teacher_weights(self):
        source = summaries()
        sample = HierarchicalGraphAggregator(config('mpq_graph', 'sample')).aggregate(source)
        quality = HierarchicalGraphAggregator(config('mpq_graph', 'sample_quality')).aggregate(source)
        torch.testing.assert_close(sample['attention_matrix'], quality['attention_matrix'])
        torch.testing.assert_close(sample['teacher_weights'], torch.tensor([.25, .75]))
        torch.testing.assert_close(quality['teacher_weights'], torch.tensor([.4, .6]))
        # The existing parameter signature includes integer state tensors too.
        # Signatures: [0,0,0,2,0,2] and [10,0,10,7,0,7]. Prototypes are orthogonal.
        off_diagonal = 7.0 / (149.0 ** .5) + .2 * .3 + .1 * .5
        diagonal = 1.0 + .35 + .35 + .2 + .1 + .3
        expected_a = torch.softmax(torch.tensor([[diagonal, off_diagonal],
                                                [off_diagonal, diagonal]]) / .5, dim=1)
        torch.testing.assert_close(sample['attention_matrix'], expected_a)
        for result in (sample, quality):
            a, omega, q = result['attention_matrix'], result['teacher_weights'], result['source_weights']
            torch.testing.assert_close(q, omega @ a)
            torch.testing.assert_close(a.sum(dim=1), torch.ones(2))
            self.assertAlmostEqual(q.sum().item(), 1.0)
            torch.testing.assert_close(result['global_parameters'][0], q[1:] * 10.)
            for i, name in enumerate(result['active_edge_ids']):
                torch.testing.assert_close(result['personalized_parameters'][name][0], a[i, 1:] * 10.)
            self.assertEqual(result['personalized_parameters']['Edge1'][1].item(), 7)

    def test_mp_excludes_quality_and_model_excludes_prototypes(self):
        source, changed = summaries(), summaries()
        changed[0].update(pseudo_quality=0.2, labeled_ratio=.9, mean_confidence=.1)
        mp = HierarchicalGraphAggregator(config('mp_graph', 'sample'))
        torch.testing.assert_close(mp.aggregate(source)['attention_matrix'], mp.aggregate(changed)['attention_matrix'])
        full = HierarchicalGraphAggregator(config())
        self.assertFalse(torch.allclose(full.aggregate(source)['attention_matrix'],
                                        full.aggregate(changed)['attention_matrix']))
        changed[0]['prototype'].fill_(4.)
        model = HierarchicalGraphAggregator(config('model_graph', 'sample'))
        torch.testing.assert_close(model.aggregate(source)['attention_matrix'], model.aggregate(changed)['attention_matrix'])

    def test_empty_missing_prototype_and_unlabeled_fallbacks(self):
        for mode, teacher in VARIANTS.values():
            aggregator = HierarchicalGraphAggregator(config(mode, teacher))
            self.assertIsNone(aggregator.aggregate([])['global_parameters'])
            source = summaries()[:1]
            source[0].update(pseudo_quality=0., mean_confidence=0., labeled_ratio=1.)
            source[0]['prototype'].zero_()
            source[0]['prototype_counts'].zero_()
            result = aggregator.aggregate(source)
            torch.testing.assert_close(result['source_weights'], torch.ones(1))
            self.assertTrue(torch.isfinite(result['global_parameters'][0]).all())

    def test_bad_configuration_and_zero_counts_fail_explicitly(self):
        for mode, teacher in [('typo', 'sample'), ('quality_average', 'sample_quality'), ('mp_graph', 'direct')]:
            with self.assertRaises(ValueError):
                HierarchicalGraphAggregator(config(mode, teacher))
        source = summaries()
        for summary in source:
            summary['batch_size'] = 0
        with self.assertRaises(ValueError):
            HierarchicalGraphAggregator(config()).aggregate(source)

    def test_legacy_graph_and_average(self):
        old_config = config(None, None)
        legacy = HierarchicalGraphAggregator(old_config).aggregate(summaries())
        explicit = HierarchicalGraphAggregator(config()).aggregate(summaries())
        torch.testing.assert_close(legacy['global_parameters'][0], explicit['global_parameters'][0])
        old_config.models.graph.use = False
        average = HierarchicalGraphAggregator(old_config).aggregate(summaries())
        self.assertEqual(average['global_parameters'][0].item(), 5.)

    def test_ema_current_source_mass(self):
        result = HierarchicalGraphAggregator(config('quality_average', 'direct')).aggregate(summaries())
        teacher = EMA(torch.nn.Linear(1, 1, bias=False), decay=.5, algorithm='AEMA')
        teacher.model.weight.data.fill_(20.)
        teacher.update_by_parameters([result['global_parameters'][0].reshape(1, 1)])
        self.assertEqual(teacher.model.weight.item(), 13.)  # .5*20 + .5*6

    def test_standalone_average_helper(self):
        # Isolate this pure helper without importing training's optional tracking/data stack.
        module = ast.parse((ROOT / 'EdgeQGFed.py').read_text(encoding='utf-8-sig'))
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'average_edge_parameters')
        namespace = {'torch': torch, 'HierarchicalGraphAggregator': HierarchicalGraphAggregator}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<average-helper>', 'exec'), namespace)
        average = namespace['average_edge_parameters']
        for mode, expected in [('uniform', 5.), ('sample', 7.5), ('sample_quality', 6.)]:
            self.assertAlmostEqual(average(summaries(), mode)[0].item(), expected)

    def test_training_dispatch_block_active_fallback_and_decay(self):
        # Execute the actual orchestration block, with synthetic summaries and no edge
        # optimizer/data/tracker. This checks integration without starting training.
        module = ast.parse((ROOT / 'EdgeQGFed.py').read_text(encoding='utf-8-sig'))
        train = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == '_train')
        block = next(node for node in ast.walk(train) if isinstance(node, ast.If)
                     and ast.unparse(node.test) == 'edge_summaries is not None')
        model = torch.nn.Module()
        model.register_parameter('weight', torch.nn.Parameter(torch.tensor([20.])))
        model.register_buffer('count', torch.tensor(100, dtype=torch.long))
        teacher = EMA(model, decay=.5, dynamic_decay=True)
        teacher.update_counts = 2  # Changes decay AFTER the update, to expose ordering bugs.
        received, events = {}, []
        edges = [SimpleNamespace(name=name, set_parameters=lambda params, name=name:
                                 received.update({name: [p.clone() for p in params]}))
                 for name in ['Edge0', 'Edge1', 'Edge2']]
        for mode, teacher_mode in [('mpq_graph', 'sample_quality'), ('quality_average', 'direct')]:
            with self.subTest(mode=mode):
                teacher.model.weight.data.fill_(20.)
                teacher.decay, teacher.update_counts = .5, 2
                aggregator = HierarchicalGraphAggregator(config(mode, teacher_mode))
                expected = aggregator.aggregate(summaries())
                namespace = {
                    'tree': SimpleNamespace(root=SimpleNamespace(model=teacher, children=edges)),
                    'edge_summaries': summaries(), 'graph_aggregator': aggregator,
                    'current_steps': 0, 'train_metrics': {}, 'cloud_downlink_mb_by_edge': {},
                    'cloud_downlink_mb': 0.,
                    'parameter_list_megabytes': lambda params, config=None: sum(p.numel()*p.element_size() for p in params)/1024**2,
                    'config': config(mode, teacher_mode), 'strict_budget_enabled': strict_budget_enabled,
                    'e1_logger': SimpleNamespace(record_aggregation=lambda *args: events.append(args)),
                }
                exec(compile(ast.Module(body=[block], type_ignores=[]), '<dispatch-block>', 'exec'), namespace)
                for edge in edges:
                    desired = expected['personalized_parameters'].get(edge.name, expected['global_parameters'])
                    torch.testing.assert_close(received[edge.name][0], desired[0])
                torch.testing.assert_close(teacher.model.weight, .5*20 + .5*expected['global_parameters'][0])
                self.assertEqual(events[-1][4], .5)
                self.assertAlmostEqual(teacher.decay, .9)
                self.assertEqual(set(namespace['cloud_downlink_mb_by_edge']), {'Edge0', 'Edge1', 'Edge2'})


class ConfigAndLogTests(unittest.TestCase):
    def test_generator_cli_all_seven_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments = ['prepare_e1_configs', '--base', str(ROOT/'config/100_basic_config.yaml'),
                         '--output-dir', temporary, '--seeds', '0', '--all-variants']
            with patch('sys.argv', arguments):
                prepare_configs()
            files = list(Path(temporary).glob('*.yaml'))
            self.assertEqual(len(files), 7)
            contents = {path.name: path.read_bytes() for path in files}
            with patch('sys.argv', arguments), self.assertRaises(SystemExit) as error:
                prepare_configs()
            self.assertEqual(error.exception.code, 2)
            self.assertEqual({path.name: path.read_bytes() for path in files}, contents)

    def test_all_configs_only_change_experiment_fields(self):
        base = OmegaConf.load(ROOT / 'config/100_basic_config.yaml')
        original = OmegaConf.to_container(base, resolve=True)
        for variant in VARIANTS:
            cfg = build_variant(base, variant, 7)
            HierarchicalGraphAggregator(cfg)
            self.assertEqual(cfg.models.ema_decay, base.models.ema_decay)
            self.assertEqual(cfg.topology, base.topology)
            self.assertEqual(cfg.train.pseudo, base.train.pseudo)
            restored = OmegaConf.to_container(cfg, resolve=True)
            restored.pop('experiment')
            restored['models']['graph'] = original['models']['graph']
            restored['datasets']['seed'] = original['datasets']['seed']
            restored['datasets']['edge_num_workers'] = original['datasets']['edge_num_workers']
            restored['train']['ckpt_save_name'] = original['train']['ckpt_save_name']
            self.assertEqual(restored, original)
        self.assertEqual(OmegaConf.to_container(base, resolve=True), original)
        self.assertEqual(len(CORE_VARIANTS), 4)

    def test_keyed_random_inputs_ignore_other_consumption_and_restore(self):
        random.seed(23)
        np.random.seed(23)
        torch.manual_seed(23)
        states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        def draw(phase):
            with e1_random_stream(7, phase, 12):
                return (random.random(), np.random.rand(), torch.rand(3))
        first = draw('edge_step')
        self.assertEqual(random.getstate(), states[0])
        np.testing.assert_equal(np.random.get_state(), states[1])
        torch.testing.assert_close(torch.get_rng_state(), states[2])
        random.random()
        np.random.rand(100)
        torch.rand(31)
        second = draw('edge_step')
        self.assertEqual(first[:2], second[:2])
        torch.testing.assert_close(first[2], second[2])
        self.assertNotEqual(draw('selection')[0], first[0])

    def test_logs_order_decay_empty_event_and_unique_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = OmegaConf.load(ROOT / 'config/100_basic_config.yaml')
            cfg = build_variant(base, 'full_sample_quality', 0, temporary)
            aggregator = HierarchicalGraphAggregator(cfg)
            logger = E1RunLogger(cfg, aggregator)
            duplicate = E1RunLogger(copy.deepcopy(cfg), aggregator)
            self.assertNotEqual(logger.directory, duplicate.directory)
            result = aggregator.aggregate(summaries())
            logger.record_aggregation(0, result, summaries(), 'AEMA', .5, ['Edge0', 'Edge1', 'Edge2'])
            logger.record_aggregation(50, aggregator.aggregate([]), [], 'AEMA', .6, ['Edge0'])
            records = [json.loads(line) for line in
                       (logger.directory / 'aggregation_events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(records[0]['active_edge_ids'], ['Edge0', 'Edge1'])
            torch.testing.assert_close(torch.tensor(records[0]['current_teacher_source_mass']),
                                       .5 * result['source_weights'])
            self.assertEqual(records[0]['dispatch']['Edge2'], 'global_aggregate')
            self.assertEqual(records[1]['dispatch'], {})
            self.assertEqual(records[1]['teacher_version_before'], records[1]['teacher_version_after'])
            # FedAvg copies the new aggregate even when the stored decay is nonzero.
            logger.record_aggregation(100, result, summaries(), 'FedAvg', .9, ['Edge0'])
            final = json.loads((logger.directory / 'aggregation_events.jsonl').read_text(encoding='utf-8').splitlines()[-1])
            self.assertEqual(final['mu_used'], 0.)


if __name__ == '__main__':
    unittest.main()

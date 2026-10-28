"""Synthetic E3 checks: no trackers, datasets, downloads, or experiment training."""

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

from datasets.parallel_clients import ParallelClientBatchLoader
from models.graph_aggregator import HierarchicalGraphAggregator
from tools.prepare_e3_configs import METHODS, build_variant, main
from utils.e1_logging import E1RunLogger
from utils.e3_stress import ValidationMixProbe, apply_label_stress, validate_stress_config
from test_e4_controls import cfg_for, make_tree, edge_helpers


ROOT = Path(__file__).resolve().parents[1]


def stress_cfg(rate=.5):
    cfg = cfg_for('all_loss_average', 'rs')
    cfg.train.total_steps = 2
    cfg.stress = dict(enabled=True, source_edge='Edge0', label_corruption_rate=rate,
                      mask_seed=0, probe_samples=3, probe_every_steps=2)
    return cfg


class ProbeData(torch.utils.data.Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return dict(img=torch.ones(6)*index/10, label=index % 2)


class E3Tests(unittest.TestCase):
    def test_source_only_mask_reproducible_and_original_data_immutable(self):
        cfg = stress_cfg()
        tree = make_tree(cfg)
        original = copy.deepcopy(tree.client_registry)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            manifest = apply_label_stress(tree.client_registry, cfg, 2)
            replay = copy.deepcopy(original)
            self.assertEqual(manifest, apply_label_stress(replay, cfg, 2))
            self.assertEqual(manifest['corrupted_samples'], 1)
            self.assertEqual(random.getstate(), python_state)
            self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
            for client_id, entry in tree.client_registry.items():
                self.assertTrue(np.array_equal(entry['targets'], original[client_id]['targets']))
                self.assertTrue(torch.equal(entry['labeled_mask'], original[client_id]['labeled_mask']))
                for pos, label in entry.get('training_label_overrides', {}).items():
                    self.assertEqual(entry['edge_name'], 'Edge0')
                    self.assertTrue(bool(entry['labeled_mask'][pos]))
                    self.assertNotEqual(label, entry['targets'][entry['indices'][pos]])
            cfg.stress.label_corruption_rate = 1.
            full = copy.deepcopy(original)
            full_manifest = apply_label_stress(full, cfg, 2)
            self.assertTrue(all(change in full_manifest['changes'] for change in manifest['changes']))
        finally:
            tree.close_batch_loaders()

    def test_overrides_reach_zero_and_spawn_client_batches_and_training(self):
        cfg = stress_cfg(1.)
        tree = make_tree(cfg)
        try:
            apply_label_stress(tree.client_registry, cfg, 2)
            results = []
            for workers in (0, 2):
                loader = ParallelClientBatchLoader(list(tree.client_registry.values()), 8, 7, workers=workers)
                try:
                    loader.prepare(tree.edge_to_clients, 0)
                    batches = {edge: loader.get_edge_batch(clients) for edge, clients in tree.edge_to_clients.items()}
                    results.append(batches)
                    # Every clean label in this fixture is zero; only source labeled items flip to one.
                    self.assertTrue((batches['Edge0']['label'][batches['Edge0']['is_labeled']] == 1).all())
                    self.assertTrue((batches['Edge1']['label'][batches['Edge1']['is_labeled']] == 0).all())
                finally:
                    loader.close()
            for edge in results[0]:
                for key in results[0][edge]:
                    self.assertTrue(torch.equal(results[0][edge][key], results[1][edge][key]))
            tree.close_batch_loaders()
            tree.init_edge_batch_loaders(cfg)
            chosen = tree.sample_clients(cfg, True, step=0)
            summaries, metrics, _ = edge_helpers()['edge_run_loop'](
                tree, chosen, torch.nn.Identity(), cfg, 0, .4,
                torch.nn.CrossEntropyLoss(), torch.nn.MSELoss(), torch.device('cpu'), update=True)
            self.assertGreater(metrics['avg_class_loss'], 0)
            source = next(s for s in summaries if s['name'] == 'Edge0')
            self.assertEqual(float(source['prototype_counts'][0]), 0.)
        finally:
            tree.close_batch_loaders()

    def test_probe_restores_modes_buffers_rng_and_records_single_step_counterfactual(self):
        cfg = stress_cfg()
        tree = make_tree(cfg)
        encoder = torch.nn.Sequential(torch.nn.Linear(6,6), torch.nn.BatchNorm1d(6))
        encoder.train()
        encoder[0].eval()  # Mixed module modes must survive the observational probe.
        modes = [m.training for m in encoder.modules()]
        state = copy.deepcopy(encoder.state_dict())
        rng = torch.get_rng_state()
        try:
            probe = ValidationMixProbe(cfg, ProbeData(), encoder, torch.device('cpu'))
            self.assertEqual(modes, [m.training for m in encoder.modules()])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(value, encoder.state_dict()[key]) for key,value in state.items()))
            self.assertFalse(probe.due(0))
            self.assertTrue(probe.due(1))
            before = probe.evaluate(tree)
            with torch.no_grad():
                tree.root.children[1].model.bias.copy_(torch.tensor([-10.,10.]))
            after = probe.evaluate(tree)
            A = torch.tensor([[.8,.2],[.1,.9]])
            omega = torch.tensor([.25,.75])
            result = dict(active_edge_ids=['Edge0','Edge1'], attention_matrix=A, source_weights=omega@A)
            with tempfile.TemporaryDirectory() as temporary:
                cfg.experiment.output_dir = temporary
                logger = E1RunLogger(cfg, HierarchicalGraphAggregator(cfg))
                probe.record(logger, 1, before, after, result)
                row = json.loads((logger.directory/'stress_mix_probes.jsonl').read_text())
                self.assertAlmostEqual(row['rows'][1]['designated_source_parameter_mass'], .1)
                self.assertEqual(row['rows'][1]['delta_loss'], after['Edge1']['loss']-before['Edge1']['loss'])
                self.assertIn('one-step no-mix', row['reference'])
        finally:
            tree.close_batch_loaders()

    def test_source_mass_is_omega_A_and_ema_adjusted_not_diagonal(self):
        cfg = stress_cfg()
        A = torch.tensor([[.8,.2],[.1,.9]])
        omega = torch.tensor([.25,.75])
        result = dict(active_edge_ids=['Edge0','Edge1'], attention_matrix=A, teacher_weights=omega,
                      source_weights=omega@A, mode='mpq_graph', teacher_weight_mode='sample_quality')
        with tempfile.TemporaryDirectory() as temporary:
            cfg.experiment.output_dir = temporary
            logger = E1RunLogger(cfg, HierarchicalGraphAggregator(cfg))
            logger.record_aggregation(0, result, [], 'AEMA', .5, ['Edge0','Edge1'])
            row = json.loads((logger.directory/'aggregation_events.jsonl').read_text())
            self.assertAlmostEqual(row['stress_source_q'], .275)
            self.assertAlmostEqual(row['stress_source_current_teacher_mass'], .1375)
            result.update(active_edge_ids=['Edge1'], attention_matrix=torch.ones(1,1),
                          teacher_weights=torch.ones(1), source_weights=torch.ones(1))
            logger.record_aggregation(1, result, [], 'FedAvg', 0, ['Edge0','Edge1'])
            rows = [json.loads(line) for line in (logger.directory/'aggregation_events.jsonl').read_text().splitlines()]
            self.assertEqual(rows[1]['stress_source_q'], 0)
            self.assertFalse(rows[1]['stress_source_active'])

    def test_generator_preserves_task_protocol_eight_configs_and_refuses_overwrite(self):
        from tools.prepare_e4_configs import build_variant as e4_variant
        base = e4_variant(OmegaConf.load(ROOT/'config/formated_config.yaml'), 'full', 'rs', 0, 'unused')
        for method in METHODS:
            cfg = build_variant(base, method, .2, 0, 'unused')
            self.assertEqual(cfg.models.ema_decay, base.models.ema_decay)
            self.assertEqual(cfg.topology, base.topology)
            self.assertEqual(cfg.train.losses, base.train.losses)
            self.assertEqual(cfg.datasets, base.datasets)
            HierarchicalGraphAggregator(cfg)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'base.yaml'
            OmegaConf.save(base, path)
            output = Path(temporary)/'output'
            args = ['e3','--base',str(path),'--output-dir',str(output)]
            with patch('sys.argv', args):
                main()
            self.assertEqual(len(list(output.glob('*.yaml'))), 8)
            with patch('sys.argv', args), patch('sys.stderr'), self.assertRaises(SystemExit):
                main()

    def test_unsafe_or_unimplemented_combined_conditions_rejected(self):
        for key, value in [('network.mobility.mode','mobility'),('models.attack_rate',.2),
                           ('train.differential.use',True),('datasets.balanced_labeled_batch_fraction',.5),
                           ('stress.source_edge','Edge99'),('stress.label_corruption_rate',float('nan'))]:
            cfg = stress_cfg()
            OmegaConf.update(cfg,key,value,force_add=True)
            with self.assertRaises(ValueError):
                validate_stress_config(cfg)


if __name__ == '__main__':
    unittest.main()

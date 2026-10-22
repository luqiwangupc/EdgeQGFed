"""No downloads/GPU/tracker: verify serial-v2 versus actual spawn workers."""

import ast
import random
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.parallel_clients import MODEL_BATCH_KEYS, ParallelClientBatchLoader
from datasets.SemiDatasets import _build_augmentations, build_base_transform
from tree.tree import Tree, TreeNode
from utils.e1_random import e1_random_stream


class RandomView:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, tensor):
        return tensor + self.scale * (torch.rand_like(tensor) + float(np.random.rand()) + random.random())


def entries():
    data = np.arange(24 * 6, dtype=np.float32).reshape(24, 6)
    labels = np.arange(24) % 2
    result = []
    for i, indices in enumerate((range(0, 4), range(4, 9), range(9, 17), range(17, 24))):
        indices = np.asarray(list(indices), dtype=np.int64)
        result.append(dict(
            client_id=f'Client{i}', edge_id=i // 2, edge_name=f'Edge{i // 2}',
            indices=indices, data=data, targets=labels, data_name='harbox',
            labeled_mask=torch.arange(len(indices)) % 2 == 0,
            transform=None, weak_transform=RandomView(.1), strong_transform=RandomView(.3),
            num_samples=len(indices),
        ))
    return result


def make_tree(workers):
    tree = Tree('Cloud')
    tree.root.children = [SimpleNamespace(name='Edge0'), SimpleNamespace(name='Edge1')]
    tree.config = OmegaConf.create({
        'datasets': {'seed': 7}, 'topology': {'client_drop_rate': .2},
        'network': {'mobility': {'mode': 'static'}},
    })
    tree.client_registry = {entry['client_id']: entry for entry in entries()}
    tree.client_network_profile = {name: {'drop_rate': .2} for name in tree.client_registry}
    tree.client_batch_loader = ParallelClientBatchLoader(
        list(tree.client_registry.values()), 6, 7, workers=workers,
        labeled_fraction=.5, prefetch_factor=1,
    )
    return tree


class ParallelClientTests(unittest.TestCase):
    def test_actual_cifar_views_match_across_worker_counts(self):
        weak, strong = _build_augmentations('cifar10')
        image_entries = [dict(
            client_id='Client0', indices=np.arange(3),
            data=np.random.default_rng(7).integers(0, 256, size=(3, 32, 32, 3), dtype=np.uint8),
            targets=np.arange(3), data_name='cifar10', labeled_mask=torch.tensor([True, False, True]),
            transform=build_base_transform('cifar10'), weak_transform=weak, strong_transform=strong,
        )]
        serial = ParallelClientBatchLoader(image_entries, 2, 7, workers=0)
        parallel = ParallelClientBatchLoader(image_entries, 2, 7, workers=1)
        try:
            groups = {'Edge0': ['Client0']}
            serial.prepare(groups, 3)
            parallel.prepare(groups, 3)
            a, b = serial.get_edge_batch(groups['Edge0']), parallel.get_edge_batch(groups['Edge0'])
            for key in MODEL_BATCH_KEYS:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        finally:
            serial.close()
            parallel.close()

    def test_spawn_matches_serial_rounds_order_and_parent_rng(self):
        torch.set_num_threads(2)
        serial = ParallelClientBatchLoader(entries(), 6, 7, workers=0, labeled_fraction=.5)
        parallel = ParallelClientBatchLoader(entries(), 6, 7, workers=2, labeled_fraction=.5)
        try:
            for step, groups in enumerate((
                {'Edge0': ['Client0', 'Client1'], 'Edge1': ['Client2', 'Client3']},
                {'Edge0': ['Client1'], 'Edge1': ['Client3', 'Client2']},
            )):
                serial.prepare(groups, step)
                parallel.prepare(groups, step)
                self.assertEqual(serial.batch_plan, parallel.batch_plan)
                self.assertEqual(len(parallel.loader._iterator._workers), 2)
                for ids in groups.values():
                    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
                    a, b = serial.get_edge_batch(ids), parallel.get_edge_batch(ids)
                    self.assertEqual(random.getstate(), states[0])
                    np.testing.assert_equal(np.random.get_state(), states[1])
                    torch.testing.assert_close(torch.get_rng_state(), states[2], rtol=0, atol=0)
                    for key in MODEL_BATCH_KEYS:
                        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
                    self.assertEqual(a['img'].shape[0], len(ids) * 6)
                self.assertEqual(serial.remaining, [])
                self.assertEqual(parallel.remaining, [])
            workers = list(parallel.loader._iterator._workers)
        finally:
            serial.close()
            parallel.close()
        self.assertTrue(all(not worker.is_alive() for worker in workers))

    def test_keyed_drop_and_empty_rounds(self):
        a, b = make_tree(0), make_tree(2)
        sampled = {'Edge0': ['Client0', 'Client1'], 'Edge1': ['Client2', 'Client3']}
        try:
            for step in (0, 1):
                a.prepare_client_batches(sampled, step)
                b.prepare_client_batches(sampled, step)
                self.assertEqual(a.active_round_clients, b.active_round_clients)
                for name, ids in a.active_round_clients.items():
                    first, second = a.get_edge_batch(ids), b.get_edge_batch(ids)
                    if ids:
                        for key in MODEL_BATCH_KEYS:
                            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
                    else:
                        self.assertIsNone(first)
                        self.assertIsNone(second)
            a.prepare_client_batches({}, 2)
            self.assertEqual(a.client_batch_loader.batch_plan, {})
        finally:
            a.close_batch_loaders()
            b.close_batch_loaders()

    def test_same_model_update_from_serial_and_spawn_inputs(self):
        # A tiny synthetic forward/backward checks the actual downstream consequence;
        # this is not an experiment run and does not construct the image encoder.
        serial = ParallelClientBatchLoader(entries(), 6, 7, workers=0)
        parallel = ParallelClientBatchLoader(entries(), 6, 7, workers=1)
        groups = {'Edge0': ['Client0', 'Client1']}
        try:
            serial.prepare(groups, 4)
            parallel.prepare(groups, 4)
            initial = torch.nn.Sequential(torch.nn.Linear(6, 2), torch.nn.Dropout(.2)).state_dict()
            states = []
            for loader in (serial, parallel):
                model = torch.nn.Sequential(torch.nn.Linear(6, 2), torch.nn.Dropout(.2))
                model.load_state_dict(initial)
                batch = loader.get_edge_batch(groups['Edge0'])
                with e1_random_stream(7, 'edge_step', 4):
                    loss = torch.nn.functional.cross_entropy(model(batch['strong_img']), batch['label'])
                    loss.backward()
                for parameter in model.parameters():
                    parameter.data.add_(parameter.grad, alpha=-.01)
                states.append(model.state_dict())
            for name in states[0]:
                torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)
        finally:
            serial.close()
            parallel.close()

    def test_actual_edge_loop_same_loss_parameters_and_payload(self):
        source = ast.parse((Path(__file__).resolve().parents[1] / 'EdgeQGFed.py').read_text(encoding='utf-8-sig'))
        names = {'edge_run_loop', 'scalar_zero', 'to_number', 'compute_consistency_loss',
                 'compute_soft_pseudo_weights', 'compute_weighted_cross_entropy',
                 'compute_labeled_prototypes', 'tensor_megabytes', 'client_payload_megabytes',
                 'edge_summary_megabytes'}
        functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = dict(torch=torch, F=torch.nn.functional, random=random, OmegaConf=OmegaConf,
                         OrderedDict=OrderedDict, get_n_classes=lambda name: 2)
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<edge-loop>', 'exec'), namespace)
        cfg = OmegaConf.create({
            'datasets': {'name': 'harbox'}, 'models': {'attack_rate': 0.},
            'network': {'client_feature_dim': 6},
            'train': {'differential': {'use': False}, 'consis_fn': 'MSE',
                      'consistency_confidence_gate': True,
                      'pseudo': {'temperature': 1., 'weight_temperature': .1,
                                 'edge_threshold': .55, 'cloud_threshold': .7,
                                 'min_weight': .1, 'start_step': 0, 'use': True, 'weight': .25}},
        })
        trees, results = [], []
        sampled = {'Edge0': ['Client0', 'Client1'], 'Edge1': ['Client2', 'Client3']}
        try:
            for workers in (0, 1):
                tree = make_tree(workers)
                trees.append(tree)
                torch.manual_seed(91)
                tree.root.model = torch.nn.Linear(6, 2)
                tree.root.children = []
                for name in sampled:
                    edge = TreeNode(name)
                    tree.root.add_child(edge)
                    edge.model = torch.nn.Sequential(torch.nn.Linear(6, 2), torch.nn.Dropout(.2))
                    edge.set_optimizer(torch.optim.SGD(edge.model.parameters(), lr=.001))
                with e1_random_stream(7, 'edge_step', 0):
                    result = namespace['edge_run_loop'](
                        tree, sampled, torch.nn.Identity(), cfg, 0, .1,
                        torch.nn.CrossEntropyLoss(), torch.nn.MSELoss(), torch.device('cpu'), update=True,
                    )
                results.append(result)
            for a, b in zip(results[0][0], results[1][0]):
                for first, second in zip(a['parameters'], b['parameters']):
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
                for key in ('prototype', 'prototype_counts'):
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            self.assertEqual(results[0][1], results[1][1])
            self.assertEqual(results[0][2], results[1][2])
        finally:
            for tree in trees:
                tree.close_batch_loaders()

    def test_plan_validation_and_legacy_loader_untouched(self):
        loader = ParallelClientBatchLoader(entries(), 6, 7, workers=0)
        try:
            loader.prepare({'Edge0': ['Client0', 'Client1']}, 0)
            with self.assertRaises(RuntimeError):
                loader.get_edge_batch(['Client1'])
            with self.assertRaises(RuntimeError):
                loader.prepare({}, 1)
            loader.get_edge_batch(['Client0', 'Client1'])
            loader.prepare({}, 1)
            self.assertIsNone(loader.get_edge_batch([]))
            with self.assertRaises(ValueError):
                loader.prepare({'Edge0': ['Client0', 'Client0']}, 2)
        finally:
            loader.close()
        tree = Tree('Cloud')
        tree.edge_to_clients = {'Edge0': ['Client0', 'Client1']}
        tree.client_registry = {entry['client_id']: entry for entry in entries()}
        cfg = OmegaConf.create({'datasets': {
            'edge_num_workers': 0, 'edge_prefetch_factor': 1,
            'batch_size': 6, 'pin_memory': False,
        }})
        tree.init_edge_batch_loaders(cfg)
        self.assertIsNone(tree.client_batch_loader)
        self.assertEqual(tree.get_edge_batch(['Client0', 'Client1'])['img'].shape[0], 12)


if __name__ == '__main__':
    unittest.main()

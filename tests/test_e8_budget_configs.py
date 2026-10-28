"""E8 CPU metadata/selector checks. No benchmark data or encoder loading."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from models.classifier import get_classifier
from models.graph_aggregator import HierarchicalGraphAggregator
from tools.prepare_e4_configs import build_variant as e4_variant
from tools.prepare_e8_configs import METHODS, build_variant, calibrate_budgets, main
from tree.tree import Tree, TreeNode, resolve_aggregation_mode
from utils.e1_random import e1_random_stream


ROOT = Path(__file__).resolve().parents[1]


def base_config():
    cfg = OmegaConf.load(ROOT/'config/harbox_config.yaml')
    cfg.datasets.val_ratio = .1
    return e4_variant(cfg, 'full', 'rs', 0, 'unused', harbox_train_users=100)


def synthetic_network_tree(cfg):
    tree = Tree('Cloud')
    tree.config = cfg
    tree.edge_to_clients = {f'Edge{i}': [] for i in range(cfg.topology.num_edges)}
    for edge_name in tree.edge_to_clients:
        node = TreeNode(edge_name)
        node.model = get_classifier(cfg.models.classifier_name, cfg.models.encoder_name, cfg.datasets.name)
        node.aggregation_mode = resolve_aggregation_mode(cfg)
        tree.root.add_child(node)
    for i in range(cfg.topology.num_clients):
        client, edge = f'Client{i}', f'Edge{i % cfg.topology.num_edges}'
        tree.edge_to_clients[edge].append(client)
        tree.client_registry[client] = dict(edge_name=edge, num_samples=100,
                                          labeled_mask=torch.tensor([True]+[False]*9))
    tree.init_network_profiles(cfg)
    return tree


class E8Tests(unittest.TestCase):
    def test_calibration_uses_actual_state_buffers_and_does_not_consume_rng(self):
        base = base_config()
        before = deepcopy(OmegaConf.to_container(base, resolve=True))
        rng = torch.get_rng_state()
        calibration = calibrate_budgets(base)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(OmegaConf.to_container(base, resolve=True), before)
        self.assertEqual(calibration['client_batch_bytes'], 16*(3*900*4+9))
        model = get_classifier('har-cnn','identity','harbox')
        self.assertEqual(calibration['model_bytes'], sum(t.numel()*t.element_size() for t in model.state_dict().values()))
        self.assertGreater(calibration['model_bytes'], sum(t.numel()*t.element_size() for t in model.parameters()))
        self.assertEqual(calibration['nominal_max_clients'], 50)
        self.assertEqual([p['nominal_client_cap'] for p in calibration['points'].values()], [13,25,50])

    def test_same_selector_traces_across_methods_and_real_budget_binding(self):
        base = base_config()
        calibration = calibrate_budgets(base)
        traces = {}
        counts = {}
        for point, meta in calibration['points'].items():
            for method in METHODS:
                cfg = build_variant(base, method, point, calibration, 0, 'unused')
                tree = synthetic_network_tree(cfg)
                with e1_random_stream(0, 'selection', 0):
                    chosen = tree.sample_clients(cfg, True, step=0)
                number = sum(map(len, chosen.values()))
                self.assertGreater(number, 0)
                self.assertLessEqual(number, meta['nominal_client_cap'])
                self.assertLessEqual(tree.selection_state['reserved_payload_bytes'], meta['budget_bytes'])
                self.assertEqual(round(tree._estimate_edge_sync_mb()*1e6), calibration['model_bytes'])
                traces[(point,method)] = (chosen, tree.selection_state)
                counts[point] = number
            self.assertEqual(traces[(point,'full')], traces[(point,'mp')])
            self.assertEqual(traces[(point,'full')], traces[(point,'all_loss_average')])
        self.assertLess(counts['b025'], counts['b050'])
        self.assertLess(counts['b050'], counts['b100'])
        self.assertEqual(counts['b100'],50)

    def test_configs_only_change_budget_aggregation_seed_and_record_fields(self):
        base = base_config()
        calibration = calibrate_budgets(base)
        for method in METHODS:
            for point in calibration['points']:
                cfg = build_variant(base,method,point,calibration,0,'unused')
                HierarchicalGraphAggregator(cfg)
                self.assertEqual(cfg.datasets, base.datasets)
                self.assertEqual(cfg.topology,base.topology)
                self.assertEqual(cfg.models.ema_decay,base.models.ema_decay)
                self.assertEqual(cfg.models.algorithm,base.models.algorithm)
                self.assertEqual(cfg.train.losses,base.train.losses)
                self.assertEqual(cfg.train.pseudo,base.train.pseudo)
                self.assertEqual(cfg.train.total_steps,base.train.total_steps)
                self.assertEqual(cfg.network.bandwidth,base.network.bandwidth)
                self.assertEqual(cfg.network.target_accuracy,base.network.target_accuracy)
                self.assertEqual(cfg.datasets.harbox_train_users,100)

    def test_nine_configs_report_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)/'base.yaml'
            destination = Path(temporary)/'output'
            OmegaConf.save(base_config(),base)
            argv=['e8','--base',str(base),'--output-dir',str(destination)]
            with patch('sys.argv',argv):
                main()
            self.assertEqual(len(list(destination.glob('*.yaml'))),9)
            self.assertEqual(len(list(destination.glob('*_s1.yaml'))),0)
            report = json.loads((destination/'budget_calibration.json').read_text())
            self.assertEqual(report['unit'],'decimal MB')
            with patch('sys.argv',argv),patch('sys.stderr'),self.assertRaises(SystemExit):
                main()

    def test_rejects_stress_legacy_and_ineffective_tiny_grids(self):
        for key,value in [('stress.enabled',True),('network.budget.strict',False),
                          ('topology.client_sample_mode','balanced'),('network.mobility.mode','mobility'),
                          ('train.losses.pseudo',False),('network.bandwidth.mode','sampled_distribution')]:
            cfg = base_config()
            OmegaConf.update(cfg,key,value,force_add=True)
            with self.assertRaises(ValueError):
                calibrate_budgets(cfg)
        cfg = base_config()
        cfg.topology.clients_per_round = 2
        with self.assertRaises(ValueError):
            calibrate_budgets(cfg)
        cfg = base_config()
        cfg.network.client_feature_dim = 900
        cfg.models.encoder_name = 'efficientv2s'
        with self.assertRaises(ValueError):
            calibrate_budgets(cfg)


if __name__ == '__main__':
    unittest.main()

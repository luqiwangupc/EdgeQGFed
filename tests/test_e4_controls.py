"""E4 synthetic CPU checks: no downloads, image encoders, trackers or experiment training."""

import ast
from collections import OrderedDict
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

from models.graph_aggregator import HierarchicalGraphAggregator
from models.ema import EMA
from tools.prepare_e4_configs import METHODS, SELECTORS, build_variant, main, validate_base
from tree.tree import Tree, TreeNode
from utils.communication import communication_unit_bytes, strict_budget_enabled
from utils.e1_logging import E1RunLogger
from utils.e1_random import e1_random_stream


ROOT = Path(__file__).resolve().parents[1]


def cfg_for(method='full', selector='random'):
    cfg = OmegaConf.create({
        'datasets': {'name': 'harbox', 'seed': 7, 'batch_size': 2, 'val_ratio': .1,
                     'pin_memory': False, 'client_batch_workers': 0, 'edge_num_workers': 0},
        'topology': {'num_clients': 4, 'num_edges': 2, 'clients_per_round': 4,
                     'edges_per_round': 2, 'client_drop_rate': 0., 'client_sample_mode': 'random_budget'},
        'network': {'client_feature_dim': 6, 'bytes_per_value': 4, 'client_view_count': 3,
                    'client_uplink_bandwidth_mbps': 10., 'max_parallel_uploads_per_edge': 2,
                    'round_comm_budget_mb': 1., 'mobility': {'mode': 'static'},
                    'budget': {'strict': True, 'include_model_sync': True}},
        'models': {'encoder_name': 'identity', 'attack_rate': 0., 'algorithm': 'AEMA',
                   'graph': {'use': True, 'mode': 'mpq_graph', 'teacher_weight_mode': 'sample_quality',
                             'temperature': .5, 'self_bias': .3, 'prototype_weight': .35,
                             'reliability_weight': .35, 'label_ratio_weight': .2,
                             'confidence_weight': .1, 'min_attention': 1e-4}},
        'train': {'ckpt_save_path': 'unused', 'ckpt_save_name': 'unused.pth',
                  'consis_fn': 'MSE', 'consistency_confidence_gate': True,
                  'final_weight': .4, 'differential': {'use': False},
                  'pseudo': {'use': True, 'weight': .25, 'start_step': 0,
                             'temperature': 1., 'weight_temperature': .1,
                             'edge_threshold': 0., 'cloud_threshold': 0., 'min_weight': .1}},
    })
    cfg = build_variant(cfg, method, selector, 7, 'unused')
    cfg.datasets.client_batch_workers = 0
    return cfg


class CountingLinear(torch.nn.Linear):
    def __init__(self):
        super().__init__(6, 2)
        self.calls = 0
        with torch.no_grad():
            self.weight.zero_()
            self.bias.copy_(torch.tensor([2., 0.]))

    def forward(self, x):
        self.calls += 1
        return super().forward(x)


def make_tree(cfg, all_unlabeled=False):
    tree = Tree('Cloud')
    tree.config = cfg
    tree.root.model = CountingLinear()
    tree.edge_to_clients = {'Edge0': [], 'Edge1': []}
    values = np.arange(8*6, dtype=np.float32).reshape(8, 6) / 100.
    for i in range(2):
        edge = TreeNode(f'Edge{i}')
        tree.root.add_child(edge)
        edge.model = CountingLinear()
        edge.set_optimizer(torch.optim.SGD(edge.model.parameters(), lr=.001))
    for i in range(4):
        name, edge = f'Client{i}', f'Edge{i//2}'
        tree.edge_to_clients[edge].append(name)
        tree.client_registry[name] = dict(
            client_id=name, edge_name=edge, data_name='harbox', num_samples=2,
            data=values, targets=np.arange(8)%2, indices=np.arange(i*2,(i+1)*2),
            labeled_mask=torch.zeros(2, dtype=torch.bool) if all_unlabeled else torch.tensor([True, False]),
            transform=None, weak_transform=torch.nn.Identity(), strong_transform=torch.nn.Identity(),
        )
        tree.client_network_profile[name] = dict(drop_rate=0., num_samples=2,
                                                 labeled_ratio=.5, client_uplink_mbps=10.+i)
    tree.init_edge_batch_loaders(cfg)
    return tree


def edge_helpers():
    module = ast.parse((ROOT/'EdgeQGFed.py').read_text(encoding='utf-8-sig'))
    names = {'edge_run_loop', 'scalar_zero', 'to_number', 'compute_consistency_loss',
             'compute_soft_pseudo_weights', 'compute_weighted_cross_entropy',
             'compute_labeled_prototypes', 'tensor_megabytes', 'client_payload_megabytes',
             'edge_summary_megabytes', 'parameter_list_megabytes'}
    functions = [node for node in module.body if isinstance(node,ast.FunctionDef) and node.name in names]
    namespace = dict(torch=torch, F=torch.nn.functional, random=random, OmegaConf=OmegaConf,
                     OrderedDict=OrderedDict, get_n_classes=lambda name: 5,
                     communication_unit_bytes=communication_unit_bytes, strict_budget_enabled=strict_budget_enabled)
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<E4-edge-helpers>','exec'),namespace)
    return namespace


class E4Tests(unittest.TestCase):
    def test_eight_configs_and_preserved_task_settings(self):
        base = OmegaConf.load(ROOT/'config/formated_config.yaml')
        validate_base(base)
        before = OmegaConf.to_container(base,resolve=True)
        for method in METHODS:
            for selector in SELECTORS:
                cfg = build_variant(base,method,selector,0,'unused')
                HierarchicalGraphAggregator(cfg)
                self.assertEqual(cfg.train.losses.consistency,METHODS[method][0])
                self.assertEqual(cfg.train.pseudo.use,METHODS[method][1])
                self.assertTrue(cfg.train.losses.supervised)
                self.assertEqual(cfg.models.ema_decay,base.models.ema_decay)
                self.assertEqual(cfg.models.algorithm,base.models.algorithm)
                self.assertEqual(cfg.datasets.partition_alpha,base.datasets.partition_alpha)
                self.assertEqual(cfg.train.pseudo.cloud_threshold,base.train.pseudo.cloud_threshold)
                self.assertEqual(cfg.train.total_steps,base.train.total_steps)
                self.assertTrue(cfg.network.budget.strict)
        self.assertEqual(OmegaConf.to_container(base,resolve=True),before)

    def test_generator_default_seed_zero_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments=['e4','--base',str(ROOT/'config/formated_config.yaml'),'--output-dir',temporary]
            with patch('sys.argv',arguments):
                main()
            self.assertEqual(len(list(Path(temporary).glob('*_s0.yaml'))),8)
            self.assertEqual(len(list(Path(temporary).glob('*_s1.yaml'))),0)
            with patch('sys.argv',arguments),patch('sys.stderr'),self.assertRaises(SystemExit):
                main()

    def test_random_has_no_utility_bias_and_exact_budget(self):
        cfg=cfg_for()
        tree=make_tree(cfg)
        try:
            self.assertAlmostEqual(tree._estimate_client_upload_mb(),162/1e6)
            full=tree.edge_to_clients
            # Two summaries: (56 model + 160 metadata), and two models per each of 2 ESs.
            self.assertAlmostEqual(tree._estimated_round_comm_mb(full,cfg,True),1304/1e6)
            cfg.network.round_comm_budget_mb=1304/1e6
            with patch.object(tree,'_client_utility',side_effect=AssertionError('random must not use utility')):
                chosen=tree.sample_clients(cfg,True,step=0)
            self.assertEqual(sum(map(len,chosen.values())),4)
            self.assertLessEqual(tree.selection_state['reserved_payload_bytes'],1304)
            cfg.network.round_comm_budget_mb=601/1e6
            self.assertEqual(tree.sample_clients(cfg,True,step=0),{})
            self.assertEqual(tree.selection_state['reserved_payload_bytes'],0)
            self.assertEqual(tree.selection_state['status'],'no_feasible_selection')
            cfg.network.round_comm_budget_mb=500/1e6
            chosen=tree.sample_clients(cfg,False,step=0)
            self.assertEqual(sum(map(len,chosen.values())),3)
        finally:
            tree.close_batch_loaders()

    def test_shared_available_opportunities_and_quota_both_selectors(self):
        cfg=cfg_for()
        cfg.topology.clients_per_round=2
        cfg.topology.edges_per_round=1
        tree=make_tree(cfg)
        try:
            for profile in tree.client_network_profile.values():
                profile['drop_rate']=.3
            opportunities=[]
            for mode in ('random_budget','resource_aware'):
                cfg.topology.client_sample_mode=mode
                with e1_random_stream(7,'selection',9):
                    chosen=tree.sample_clients(cfg,True,step=9)
                opportunities.append(tree.selection_state['available'])
                self.assertLessEqual(len(chosen),1)
                self.assertLessEqual(sum(map(len,chosen.values())),2)
                self.assertTrue(all(c in opportunities[-1] for ids in chosen.values() for c in ids))
            self.assertEqual(opportunities[0],opportunities[1])
        finally:
            tree.close_batch_loaders()

    def test_loss_branches_and_runtime_payload_match_reservation(self):
        functions=edge_helpers()
        for method in METHODS:
            cfg=cfg_for(method)
            tree=make_tree(cfg)
            try:
                chosen=tree.sample_clients(cfg,True,step=0)
                with e1_random_stream(7,'edge_step',0):
                    summaries,metrics,network=functions['edge_run_loop'](
                        tree,chosen,torch.nn.Identity(),cfg,0,.4,
                        torch.nn.CrossEntropyLoss(),torch.nn.MSELoss(),torch.device('cpu'),update=True)
                expected_calls=1 if method=='hfedavg' else 2
                self.assertTrue(all(edge.model.calls==expected_calls for edge in tree.root.children))
                self.assertEqual(tree.root.model.calls,0 if method=='hfedavg' else len(chosen))
                if not METHODS[method][0]:
                    self.assertEqual(metrics['avg_consis_loss'],0.)
                if method=='hfedavg':
                    self.assertEqual(metrics['avg_pseudo_loss'],0.)
                    self.assertTrue(all(not s['quality_observed'] for s in summaries))
                actual=(sum(network['client_upload_mb_by_edge'].values())
                        +sum(network['edge_uplink_mb_by_edge'].values())
                        +2*len(tree.root.children)*tree._estimate_edge_sync_mb())
                self.assertAlmostEqual(actual,tree.selection_state['reserved_payload_bytes']/1e6)
                self.assertEqual(network['active_client_ids_by_edge'],tree.active_round_clients)
            finally:
                tree.close_batch_loaders()

    def test_supervised_only_unlabeled_batch_does_not_backward(self):
        cfg=cfg_for('hfedavg')
        tree=make_tree(cfg,all_unlabeled=True)
        chosen=tree.sample_clients(cfg,True,step=0)
        try:
            with patch.object(tree.root.children[0],'optimizer_step',side_effect=AssertionError('empty loss')):
                _,metrics,_=edge_helpers()['edge_run_loop'](
                    tree,chosen,torch.nn.Identity(),cfg,0,.4,torch.nn.CrossEntropyLoss(),
                    torch.nn.MSELoss(),torch.device('cpu'))
            self.assertEqual(metrics['total_loss'],0.)
        finally:
            tree.close_batch_loaders()

    def test_tensor_ledger_is_integer_bytes(self):
        cfg=cfg_for()
        with tempfile.TemporaryDirectory() as temporary:
            cfg.experiment.output_dir=temporary
            logger=E1RunLogger(cfg,HierarchicalGraphAggregator(cfg))
            logger.record_communication(0,.000648,.000432,.000112,.000112,.001304,.001304,
                                        {'reserved_payload_bytes':1304})
            item=json.loads((logger.directory/'communication_by_round.jsonl').read_text())
            self.assertEqual(item['round_bytes'],1304)
            self.assertEqual(item['budget_bytes'],1304)
            self.assertEqual(item['by_direction']['teacher_to_all_edges'],112)

    def test_actual_dispatch_and_runtime_budget_guard(self):
        cfg=cfg_for()
        tree=make_tree(cfg)
        helpers=edge_helpers()
        source=ast.parse((ROOT/'EdgeQGFed.py').read_text(encoding='utf-8-sig'))
        train=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='_train')
        dispatch=next(n for n in ast.walk(train) if isinstance(n,ast.If)
                      and ast.unparse(n.test)=='edge_summaries is not None')
        guard=next(n for n in ast.walk(train) if isinstance(n,ast.If)
                   and ast.unparse(n.test)=='strict_budget_enabled(config)'
                   and any(isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                           and call.func.attr=='record_communication' for call in ast.walk(n)))
        try:
            chosen=tree.sample_clients(cfg,True,step=0)
            summaries,_,network=helpers['edge_run_loop'](
                tree,chosen,torch.nn.Identity(),cfg,0,.4,torch.nn.CrossEntropyLoss(),
                torch.nn.MSELoss(),torch.device('cpu'),update=True)
            tree.root.model=EMA(tree.root.model,decay=.5,algorithm='AEMA')
            with tempfile.TemporaryDirectory() as temporary:
                cfg.experiment.output_dir=temporary
                logger=E1RunLogger(cfg,HierarchicalGraphAggregator(cfg))
                namespace=dict(config=cfg,tree=tree,edge_summaries=summaries,
                               graph_aggregator=HierarchicalGraphAggregator(cfg),
                               strict_budget_enabled=strict_budget_enabled,
                               parameter_list_megabytes=helpers['parameter_list_megabytes'],
                               current_steps=0,train_metrics={},cloud_downlink_mb=0.,
                               cloud_downlink_mb_by_edge={},teacher_downlink_mb=0.,e1_logger=logger)
                exec(compile(ast.Module(body=[dispatch],type_ignores=[]),'<dispatch>','exec'),namespace)
                self.assertAlmostEqual(namespace['cloud_downlink_mb'],224/1e6)
                self.assertAlmostEqual(namespace['teacher_downlink_mb'],112/1e6)
                namespace.update(client_upload_mb=sum(network['client_upload_mb_by_edge'].values()),
                                 edge_upload_mb=sum(network['edge_uplink_mb_by_edge'].values()))
                namespace['total_comm_mb']=namespace['client_upload_mb']+namespace['edge_upload_mb']+namespace['cloud_downlink_mb']
                namespace['cumulative_comm_mb']=namespace['total_comm_mb']
                cfg.network.round_comm_budget_mb=namespace['total_comm_mb']
                exec(compile(ast.Module(body=[guard],type_ignores=[]),'<budget-guard>','exec'),namespace)
                item=json.loads((logger.directory/'communication_by_round.jsonl').read_text())
                self.assertEqual(item['round_bytes'],1304)
                cfg.network.round_comm_budget_mb-=.000001
                with self.assertRaises(RuntimeError):
                    exec(compile(ast.Module(body=[guard],type_ignores=[]),'<budget-guard>','exec'),namespace)
        finally:
            tree.close_batch_loaders()


if __name__=='__main__':
    unittest.main()

"""Synthetic E8 bandwidth checks: no benchmark data, encoder or experiment training."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from tools.prepare_e8_bandwidth_configs import bandwidth_scenarios, build_variant, main
from tools.prepare_e8_configs import METHODS, build_variant as budget_variant, calibrate_budgets
from utils.e1_random import e1_random_stream
from test_e8_budget_configs import base_config, synthetic_network_tree


def budget_base():
    base = base_config()
    return budget_variant(base,'full','b050',calibrate_budgets(base),0,'unused')


class BandwidthTests(unittest.TestCase):
    def test_mean_and_variance_and_links_preserved(self):
        base = budget_base()
        before = deepcopy(OmegaConf.to_container(base,resolve=True))
        scenarios = bandwidth_scenarios(base)
        self.assertEqual(scenarios['points']['h000']['client_uplink_by_edge_mbps'],[23.]*5)
        self.assertEqual(scenarios['points']['h050']['client_uplink_by_edge_mbps'],[14.,16.5,21.5,26.5,36.5])
        self.assertEqual(scenarios['points']['h100']['client_uplink_by_edge_mbps'],[5.,10.,20.,30.,50.])
        self.assertAlmostEqual(scenarios['points']['h050']['all_client_cv']*2,scenarios['points']['h100']['all_client_cv'])
        for point in scenarios['points']:
            cfg=build_variant(base,'mp',point,scenarios,0,'unused')
            self.assertEqual(cfg.datasets,base.datasets)
            self.assertEqual(cfg.topology,base.topology)
            self.assertEqual(cfg.train,base.train)
            self.assertEqual(cfg.models.algorithm,base.models.algorithm)
            self.assertEqual(cfg.models.ema_decay,base.models.ema_decay)
            self.assertEqual(cfg.network.round_comm_budget_mb,base.network.round_comm_budget_mb)
            tree=synthetic_network_tree(cfg)
            self.assertAlmostEqual(sum(p['client_uplink_mbps'] for p in tree.client_network_profile.values())/100,23.)
            self.assertEqual([p['edge_uplink_mbps'] for p in tree.edge_network_profile.values()],[80.,120.,200.,300.,400.])
            self.assertEqual([p['cloud_downlink_mbps'] for p in tree.edge_network_profile.values()],[150.,250.,400.,500.,600.])
        self.assertEqual(before,OmegaConf.to_container(base,resolve=True))

    def test_population_weighting_for_unequal_edge_populations(self):
        base=budget_base()
        base.topology.num_clients=101
        scenarios=bandwidth_scenarios(base)
        expected=(21*5+20*(10+20+30+50))/101
        self.assertAlmostEqual(scenarios['all_client_mean_mbps'],expected)
        self.assertEqual(scenarios['clients_by_edge'],[21,20,20,20,20])
        for point in scenarios['points'].values():
            self.assertAlmostEqual(point['all_client_mean_mbps'],expected)

    def test_same_condition_method_selection_matches_and_h100_replays_budget(self):
        base=budget_base()
        scenarios=bandwidth_scenarios(base)
        traces={}
        for point in scenarios['points']:
            for method in METHODS:
                tree=synthetic_network_tree(build_variant(base,method,point,scenarios,0,'unused'))
                with e1_random_stream(0,'selection',3):
                    selected=tree.sample_clients(tree.config,True,step=3)
                traces[(point,method)]=(selected,tree.selection_state)
            self.assertEqual(traces[(point,'full')],traces[(point,'mp')])
            self.assertEqual(traces[(point,'full')],traces[(point,'all_loss_average')])
        original=synthetic_network_tree(base)
        with e1_random_stream(0,'selection',3):
            selected=original.sample_clients(base,True,step=3)
        self.assertEqual(traces[('h100','full')],(selected,original.selection_state))
        self.assertEqual(traces[('h000','full')][1]['available'],traces[('h100','full')][1]['available'])

    def test_cli_nine_configs_no_overwrite_and_reuse_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)/'base.yaml'
            output=Path(temporary)/'output'
            OmegaConf.save(budget_base(),base)
            args=['bandwidth','--base',str(base),'--output-dir',str(output)]
            with patch('sys.argv',args):
                main()
            self.assertEqual(len(list(output.glob('*.yaml'))),9)
            cfg=OmegaConf.load(output/'full_h100_b050_s0.yaml')
            self.assertEqual(cfg.experiment.e8.budget_reuse_candidate,'full_b050_s0')
            report=json.loads((output/'bandwidth_scenarios.json').read_text())
            self.assertEqual(report['budget_point'],'b050')
            with patch('sys.argv',args),patch('sys.stderr'),self.assertRaises(SystemExit):
                main()

    def test_rejects_invalid_profiles_and_wrong_base(self):
        for key,value in [('experiment.e8.scan','client_bandwidth'),('network.bandwidth.mode','homogeneous'),
                          ('network.bandwidth.client_uplink_by_edge_mbps',[5]*5),
                          ('network.bandwidth.client_uplink_by_edge_mbps',[5,10]),
                          ('network.bandwidth.client_uplink_by_edge_mbps',[5,10,20,30,float('inf')])]:
            cfg=budget_base()
            OmegaConf.update(cfg,key,value,force_add=True)
            with self.assertRaises(ValueError):
                bandwidth_scenarios(cfg)


if __name__=='__main__':
    unittest.main()

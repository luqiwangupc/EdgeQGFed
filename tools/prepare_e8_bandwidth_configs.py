"""Generate a separate, fixed-mean E8 client-bandwidth scan at one existing budget."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf

from tools.prepare_e8_configs import METHODS, validate_e8_base


def bandwidth_scenarios(base, percentages=(0, 50, 100)):
    validate_e8_base(base)
    if OmegaConf.select(base, 'experiment.e8.scan') != 'budget':
        raise ValueError('Use the existing E8 budget YAML, not an E4 or bandwidth-scan YAML')
    if str(base.network.bandwidth.mode) != 'edge_profile':
        raise ValueError('Bandwidth scan requires the original heterogeneous edge_profile base')
    edges, clients = int(base.topology.num_edges), int(base.topology.num_clients)
    original = OmegaConf.select(base, 'network.bandwidth.client_uplink_by_edge_mbps')
    if original is None or len(original) != edges:
        raise ValueError('Provide exactly one client uplink value per ES')
    original = [float(value) for value in original]
    if any(not math.isfinite(value) or value <= 0 for value in original):
        raise ValueError('Client bandwidths must be positive finite Mbps')
    # Static round-robin association: weight by ALL clients, not selected clients.
    populations = [len(range(i, clients, edges)) for i in range(edges)]
    mean = sum(n*b for n,b in zip(populations, original))/clients
    variance = sum(n*(b-mean)**2 for n,b in zip(populations, original))/clients
    if variance <= 1e-20:
        raise ValueError('Base profile is already homogeneous; no distinct heterogeneity scan is possible')
    percentages = list(dict.fromkeys(int(value) for value in percentages))
    if not percentages or any(value < 0 or value > 100 for value in percentages):
        raise ValueError('heterogeneity-percent must be integer values in [0, 100]')
    points = {}
    for percent in percentages:
        values = original[:] if percent == 100 else [mean+(percent/100)*(b-mean) for b in original]
        actual_mean = sum(n*b for n,b in zip(populations,values))/clients
        cv = math.sqrt(sum(n*(b-actual_mean)**2 for n,b in zip(populations,values))/clients)/actual_mean
        points[f'h{percent:03d}'] = dict(heterogeneity_percent=percent,
            client_uplink_by_edge_mbps=values, all_client_mean_mbps=actual_mean,
            all_client_cv=cv, reuse_budget_condition=percent == 100)
    return dict(protocol='client_edge_profile_fixed_mean_v1',
                budget_point=str(base.experiment.e8.point), budget_mb=float(base.network.round_comm_budget_mb),
                source_client_uplink_by_edge_mbps=original, clients_by_edge=populations,
                all_client_mean_mbps=mean, points=points,
                scope='Only client uplink profile changes; ES uplink/downlink, budget, selector and opportunities remain fixed. '
                      'Mean is over all logical clients; selected-client mean may differ because RS responds to bandwidth.')


def build_variant(base, method, point_name, scenarios, seed, run_root, base_sha256=None):
    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    mode, teacher = METHODS[method]
    cfg.models.graph.mode = mode
    cfg.models.graph.use = mode.endswith('_graph')
    cfg.models.graph.teacher_weight_mode = teacher
    cfg.datasets.seed = int(seed)
    budget_point = scenarios['budget_point']
    variant = f'{method}_{point_name}_{budget_point}'
    cfg.network.bandwidth.client_uplink_by_edge_mbps = scenarios['points'][point_name]['client_uplink_by_edge_mbps']
    cfg.experiment = OmegaConf.merge(cfg.experiment, dict(
        family='e8', name=f'e8_{variant}_s{seed}', variant=variant, output_dir=str(run_root),
        base_sha256=base_sha256,
        comparison_scope='fixed-mean client-bandwidth heterogeneity at a fixed budget with the same selector policy'))
    cfg.experiment.e8 = dict(scan='client_bandwidth', point=point_name, budget_point=budget_point,
                             budget_calibration=OmegaConf.to_container(base.experiment.e8.calibration, resolve=True),
                             bandwidth_scenarios=scenarios,
                             budget_reuse_candidate=f'{method}_{budget_point}_s{seed}' if point_name == 'h100' else None,
                             pairing='Within each condition, methods must share selected/successful traces; '
                                     'across heterogeneity, availability is shared but RS selection may change.')
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='Existing clean E8 full_b050_s0.yaml budget config')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--run-root', default='runs/e8_bandwidth')
    parser.add_argument('--heterogeneity-percent', nargs='+', type=int, default=[0, 50, 100])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    args = parser.parse_args()
    path = Path(args.base)
    if not path.is_file():
        parser.error(f'Base not found: {path}; use the actual E8 budget YAML')
    try:
        base = OmegaConf.load(path)
        scenarios = bandwidth_scenarios(base,args.heterogeneity_percent)
        signature = hashlib.sha256(path.read_bytes()).hexdigest()
        destination = Path(args.output_dir)
        tasks = [(destination/f'{method}_{point}_{scenarios["budget_point"]}_s{seed}.yaml',
                  build_variant(base,method,point,scenarios,seed,args.run_root,signature))
                 for seed in dict.fromkeys(args.seeds) for point in scenarios['points'] for method in METHODS]
        report = destination/'bandwidth_scenarios.json'
        if report.exists() or any(output.exists() for output,_ in tasks):
            raise ValueError('Output exists; choose a new directory rather than overwriting previous configs')
    except (ValueError,TypeError) as error:
        parser.error(str(error))
    destination.mkdir(parents=True,exist_ok=True)
    for output,cfg in tasks:
        OmegaConf.save(cfg,output,resolve=True)
    report.write_text(json.dumps(dict(base_sha256=signature,**scenarios),indent=2),encoding='utf-8')
    print(f'Generated {len(tasks)} configs; no data, encoder, checkpoint or training loaded.')
    for name,point in scenarios['points'].items():
        print(f"{name}: client uplinks {point['client_uplink_by_edge_mbps']}, "
              f"mean={point['all_client_mean_mbps']:.6f} Mbps, CV={point['all_client_cv']:.6f}")
    print('h100 matches the original budget condition; verify corresponding runs before scheduling duplicates.')


if __name__ == '__main__':
    main()

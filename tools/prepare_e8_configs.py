"""Calibrate tensor budgets and generate E8 configs without loading data/encoders."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
import torch

from models.classifier import ENCODER_FEATURE_DIMS, get_classifier
from tools.prepare_e4_configs import validate_base
from utils.communication import summary_metadata_bytes


METHODS = {
    'full': ('mpq_graph', 'sample_quality'),
    'mp': ('mp_graph', 'sample_quality'),
    'all_loss_average': ('sample_average', 'direct'),
}


def validate_e8_base(base):
    validate_base(base)
    required = ('network.budget.strict', 'network.budget.include_model_sync',
                'experiment.paired_inputs', 'experiment.e1_logging',
                'train.losses.supervised', 'train.losses.consistency', 'train.losses.pseudo')
    if not all(bool(OmegaConf.select(base, key, default=False)) for key in required):
        raise ValueError('Use a strict E4 all-loss base with paired client inputs and local records')
    if OmegaConf.select(base, 'datasets.client_batch_workers') is None:
        raise ValueError('E8 requires the client-keyed input backend')
    if str(base.topology.client_sample_mode) not in {'resource_aware', 'random_budget'}:
        raise ValueError('E8 requires the same feasible RS or random_budget selector for all methods')
    if str(base.topology.assignment) != 'round_robin':
        raise ValueError('E8 calibration currently supports round_robin assignment only')
    if str(OmegaConf.select(base, 'network.mobility.mode', default='static')) != 'static':
        raise ValueError('First E8 budget scan requires static association/opportunity probabilities')
    if str(OmegaConf.select(base, 'network.bandwidth.mode', default='homogeneous')) not in {'homogeneous', 'edge_profile'}:
        raise ValueError('First E8 budget scan requires fixed homogeneous/edge_profile bandwidths')
    if bool(OmegaConf.select(base, 'stress.enabled', default=False)) or float(base.models.attack_rate) != 0:
        raise ValueError('Use a clean E4 base, not an E3 stress/attack configuration')
    if int(base.train.ema_update_step) < 1:
        raise ValueError('ema_update_step must be positive')
    if any(int(v) < 1 for v in (base.topology.num_clients, base.topology.num_edges,
                                base.topology.clients_per_round, base.topology.edges_per_round,
                                base.network.max_parallel_uploads_per_edge, base.datasets.batch_size)):
        raise ValueError('E8 quotas/batch size must be positive')


def calibrate_budgets(base, percentages=(25, 50, 100)):
    """Budget labels are fractions of a nominal client cap, NOT fractions of MB.

    At restricted points use the smallest possible sync envelope for k clients;
    extra active ESs can reduce achieved k. The full point covers all legal ESs.
    No claim about actual availability/data emptiness is made without run traces.
    """
    validate_e8_base(base)
    percentages = list(dict.fromkeys(int(p) for p in percentages))
    if not percentages or any(p < 1 or p > 100 for p in percentages):
        raise ValueError('client-cap-percent must be integer values in [1, 100]')
    # Only an untrained CPU classifier is built; no dataset, weights or encoder.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = get_classifier(str(base.models.classifier_name), str(base.models.encoder_name),
                               str(base.datasets.name), float(OmegaConf.select(base, 'models.dropout_rate', default=.5)),
                               int(base.network.client_feature_dim))
    aggregation = str(OmegaConf.select(base, 'models.aggregation_mode', default='auto'))
    if aggregation == 'auto':
        aggregation = 'state_dict' if str(base.datasets.name) == 'harbox' else 'parameters'
    if aggregation not in {'state_dict', 'parameters'}:
        raise ValueError('Unknown aggregation mode')
    tensors = model.state_dict().values() if aggregation == 'state_dict' else model.parameters()
    model_bytes = sum(t.numel()*t.element_size() for t in tensors)
    classes = 5 if str(base.datasets.name) == 'harbox' else 10
    dim = int(ENCODER_FEATURE_DIMS.get(str(base.models.encoder_name), base.network.client_feature_dim))
    if dim != int(base.network.client_feature_dim):
        raise ValueError('Configured feature_dim does not match the encoder')
    client_bytes = int(base.datasets.batch_size)*(dim*3*4 + 8 + 1)
    summary_bytes = model_bytes + summary_metadata_bytes(classes, dim)
    clients, edges = int(base.topology.num_clients), int(base.topology.num_edges)
    cap = int(base.network.max_parallel_uploads_per_edge)
    populations = [len(range(i, clients, edges)) for i in range(edges)]
    capacities = sorted((min(n, cap) for n in populations if n > 0), reverse=True)
    max_edges = min(len(capacities), int(base.topology.edges_per_round))
    capacities = capacities[:max_edges]
    max_clients = min(int(base.topology.clients_per_round), sum(capacities))
    max_edges = min(max_edges, max_clients)
    if max_clients < 2:
        raise ValueError('Need at least two nominal clients to create a meaningful budget scan')
    dispatch_bytes = 2*edges*model_bytes
    points = {}
    seen_budgets = set()
    for percentage in percentages:
        k = max(1, int(math.ceil(max_clients*percentage/100)))
        if percentage < 100 and k == max_clients:
            raise ValueError('Restricted point rounds to full capacity; choose a lower percentage')
        packed, min_edges = 0, 0
        for capacity in capacities:
            packed += capacity
            min_edges += 1
            if packed >= k:
                break
        active_edges = max_edges if percentage == 100 else min_edges
        payload = k*client_bytes + active_edges*summary_bytes + dispatch_bytes
        if payload in seen_budgets:
            raise ValueError('Budget points collapse to the same payload; use fewer/distinct percentages')
        seen_budgets.add(payload)
        points[f'b{percentage:03d}'] = dict(client_cap_percent=percentage,
            nominal_client_cap=k, budget_bytes=payload, budget_mb=payload/1_000_000,
            calibration_active_edges=active_edges,
            interpretation='full legal sync envelope' if percentage == 100 else
                           'upper bound on admitted clients at sync; actual admission may be lower')
    return dict(protocol='e8_sync_payload_cap_v1', unit='decimal MB',
                model_bytes=model_bytes, aggregation_mode=aggregation,
                client_batch_bytes=client_bytes, active_edge_summary_bytes=summary_bytes,
                all_edge_model_teacher_dispatch_bytes=dispatch_bytes,
                nominal_max_clients=max_clients, nominal_max_active_edges=max_edges,
                points=points,
                limits='No data/availability inspection; nominal capacities assume nonempty round-robin clients. '
                       'On non-sync rounds budget only covers client uploads. Not actual network bytes or energy.')


def build_variant(base, method, point_name, calibration, seed, run_root, base_sha256=None):
    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    mode, teacher = METHODS[method]
    cfg.models.graph.mode = mode
    cfg.models.graph.use = mode.endswith('_graph')
    cfg.models.graph.teacher_weight_mode = teacher
    cfg.datasets.seed = int(seed)
    cfg.datasets.client_batch_workers = 4
    cfg.datasets.eval_num_workers = 0
    cfg.datasets.edge_num_workers = 0
    cfg.train.cpu_threads = 2
    cfg.train.ckpt_save_name = 'best.pth'
    cfg.network.round_comm_budget_mb = calibration['points'][point_name]['budget_mb']
    cfg.experiment = OmegaConf.merge(cfg.get('experiment', {}), dict(
        family='e8', name=f'e8_{method}_{point_name}_s{seed}', variant=f'{method}_{point_name}',
        output_dir=str(run_root), e1_logging=True, paired_inputs=True, base_sha256=base_sha256,
        comparison_scope='clean budget scan under matched tensor payload and common selector; not full E6 cost'))
    cfg.experiment.e8 = dict(scan='budget', point=point_name, calibration=calibration,
                             selection_pairing='same method-independent selector RNG per condition; verify traces')
    if str(cfg.datasets.name) == 'harbox' and OmegaConf.select(cfg, 'datasets.harbox_train_users') is None:
        raise ValueError('E8 HARBox base must explicitly record datasets.harbox_train_users from E4')
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='Existing clean strict E4 full/all-loss RS YAML')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--run-root', default='runs/e8')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--client-cap-percent', nargs='+', type=int, default=[25, 50, 100])
    args = parser.parse_args()
    base_path = Path(args.base)
    if not base_path.is_file():
        parser.error(f'Base not found: {base_path}; use the actual E4 all-loss/full YAML')
    try:
        base = OmegaConf.load(base_path)
        calibration = calibrate_budgets(base, args.client_cap_percent)
        signature = hashlib.sha256(base_path.read_bytes()).hexdigest()
        directory = Path(args.output_dir)
        tasks = [(directory/f'{method}_{point}_s{seed}.yaml',
                  build_variant(base, method, point, calibration, seed, args.run_root, signature))
                 for seed in dict.fromkeys(args.seeds) for point in calibration['points'] for method in METHODS]
        report = directory/'budget_calibration.json'
        if report.exists() or any(path.exists() for path, _ in tasks):
            raise ValueError('Output already exists; choose a new directory, do not overwrite previous tasks')
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    directory.mkdir(parents=True, exist_ok=True)
    for path, cfg in tasks:
        OmegaConf.save(cfg, path, resolve=True)
    report.write_text(json.dumps(dict(base_sha256=signature, **calibration), indent=2), encoding='utf-8')
    print(f'Generated {len(tasks)} E8 configs; no data loaded, encoder constructed, or training started.')
    for name, point in calibration['points'].items():
        print(f"{name}: {point['budget_mb']:.6f} MB/sync round; nominal client cap {point['nominal_client_cap']}")
    print('b025/b050 are client-cap fractions, not 25/50 MB or 25/50% of full MB. Verify actual run traces.')


if __name__ == '__main__':
    main()

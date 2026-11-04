"""Clone actual seed-0 configs for figure repeats; never changes method/topology."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from omegaconf import OmegaConf


def describe_base(base):
    explicit = OmegaConf.select(base, 'models.graph.mode')
    graph_mode = str(explicit) if explicit is not None else (
        'mpq_graph' if bool(OmegaConf.select(base, 'models.graph.use', default=False)) else 'uniform_average')
    warnings = []
    if graph_mode != 'mpq_graph':
        warnings.append(f'Inherited graph mode is {graph_mode}, not the full quality graph; unchanged. Confirm original figure provenance.')
    if float(OmegaConf.select(base, 'datasets.val_ratio', default=0)) <= 0:
        warnings.append('No held-out validation configured: current training uses official test for validation/model selection. '
                        'Do not treat these validation curves as untouched test results. Obtain actual source config; '
                        'adding a split only for new seeds changes the protocol.')
    if OmegaConf.select(base, 'datasets.seed') is None:
        warnings.append('Source YAML did not record a seed; current runtime defaults to 0, historical run seed is unconfirmed.')
    if OmegaConf.select(base, 'datasets.client_batch_workers') is None:
        warnings.append('Client loader CLI override is not recorded. Use resolved config from the original run when possible; '
                        'do not add --client-workers only to new seeds and call it the same input protocol.')
    if str(base.datasets.name) == 'harbox' and OmegaConf.select(base, 'datasets.harbox_train_users') is None:
        warnings.append('HARBox training-user environment is unrecorded. Match original HARBOX_ROOT/TRAIN_USERS/WINDOW settings.')
    return dict(dataset=str(base.datasets.name), source_seed=OmegaConf.select(base, 'datasets.seed'),
                graph_mode=graph_mode, selector=OmegaConf.select(base, 'topology.client_sample_mode'),
                num_edges=OmegaConf.select(base, 'topology.num_edges'),
                num_clients=OmegaConf.select(base, 'topology.num_clients'),
                clients_per_round=OmegaConf.select(base, 'topology.clients_per_round'),
                edges_per_round=OmegaConf.select(base, 'topology.edges_per_round'),
                val_ratio=OmegaConf.select(base, 'datasets.val_ratio', default=0),
                client_batch_workers=OmegaConf.select(base, 'datasets.client_batch_workers'),
                warnings=warnings)


def build_repeat(base, group, seed, run_root, signature=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', group):
        raise ValueError('Group must contain only letters, digits, underscores or hyphens')
    if int(seed) not in {1, 2}:
        raise ValueError('This repeat package is restricted to seeds 1 and 2')
    if bool(OmegaConf.select(base, 'stress.enabled', default=False)):
        raise ValueError('Figures 2/3 need clean source configs, not E3 stress/probe configs')
    if int(OmegaConf.select(base, 'datasets.seed', default=0)) != 0:
        raise ValueError('Provide the actual seed-0 config as the source')
    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    cfg.datasets.seed = int(seed)
    # Only seed and local output/bookkeeping change. In particular, do NOT turn
    # on paired_inputs, a new loader, strict budgets, graph modes or new splits.
    cfg.train.ckpt_save_path = (Path(run_root)/f'{group}_s{seed}'/'checkpoints').as_posix()
    cfg.experiment = OmegaConf.merge(cfg.get('experiment', {}), dict(
        family='fig23_repeats', name=f'{group}_s{seed}', variant=group,
        e1_logging=True, output_dir=(Path(run_root)/group).as_posix(),
        repeat_source_sha256=signature, repeat_source_seed=0,
        repeat_scope='seed-only training-input change; historical figure provenance must be verified'))
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', nargs='+', required=True, metavar='GROUP=PATH',
                        help='One actual seed-0 YAML per plotted group; paths may be server-only')
    parser.add_argument('--seeds', type=int, nargs='+', choices=[1,2], default=[1,2])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--run-root', default='runs/fig23_repeats')
    args = parser.parse_args()
    try:
        groups, tasks, manifest = set(), [], []
        destination = Path(args.output_dir)
        for spec in args.base:
            if '=' not in spec:
                raise ValueError('Each --base item must be GROUP=PATH')
            group, source = spec.split('=', 1)
            if group in groups:
                raise ValueError(f'Duplicate group: {group}')
            groups.add(group)
            path = Path(source)
            if not path.is_file():
                raise ValueError(f'Actual seed-0 YAML not found for {group}: {path}')
            base = OmegaConf.load(path)
            signature = hashlib.sha256(path.read_bytes()).hexdigest()
            description = describe_base(base)
            outputs = []
            for seed in dict.fromkeys(args.seeds):
                output = destination/f'{group}_s{seed}.yaml'
                cfg = build_repeat(base,group,seed,args.run_root,signature)
                tasks.append((output,cfg))
                outputs.append(output.name)
            manifest.append(dict(group=group, source=str(path), source_sha256=signature,
                                 outputs=outputs, **description))
        report = destination/'repeat_sources.json'
        if report.exists() or any(path.exists() for path,_ in tasks):
            raise ValueError('Output exists; use a new directory to preserve previous tasks')
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    destination.mkdir(parents=True,exist_ok=True)
    for output,cfg in tasks:
        OmegaConf.save(cfg,output,resolve=True)
    report.write_text(json.dumps(dict(scope='provenance unverified until matched to original runs',
                                     groups=manifest),indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Generated {len(tasks)} YAMLs. No training, data, weights or encoder loaded.')
    for group in manifest:
        for warning in group['warnings']:
            print(f"WARNING [{group['group']}]: {warning}")


if __name__=='__main__':
    main()

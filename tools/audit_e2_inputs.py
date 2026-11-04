"""Read-only E2 inventory: stdout JSON, no torch/torchvision/trackers/downloads.

File presence is not proof of diagnostic isolation. Checkpoint files are stat'ed
only. HARBox ZIP entries are listed without reading/extracting sensor records.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

from omegaconf import OmegaConf


ACTIVITIES = {'Call', 'Hop', 'Walk', 'Wave', 'typing'}
PROJECT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_FIELDS = (
    'datasets.diagnostic_ratio', 'datasets.diagnostic_split',
    'datasets.diagnostic_manifest', 'experiment.diagnostic_manifest',
)


def metadata(path):
    if not path.is_file():
        return {'exists': False}
    return {'exists': True, 'bytes': path.stat().st_size}


def read_json(path):
    if not path.is_file():
        return None
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError('JSON metadata exceeds 2 MiB inspection limit')
    with path.open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def partition_headers(path):
    """Inspect NPY headers only; never unpickle or read sample index arrays."""
    if not path.is_file():
        return {'exists': False}
    import numpy as np
    clients = masks = 0
    rows = 0
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) > 100000:
            raise ValueError('Partition archive has too many entries')
        for entry in archive.infolist():
            if not entry.filename.endswith('.npy'):
                continue
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=16384)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=16384)
                else:
                    raise ValueError(f'Unrecognized NPY header version: {version}')
            if dtype.hasobject:
                raise ValueError('Object arrays are not accepted for audit')
            if entry.filename.endswith('_indices.npy'):
                clients += 1
                if len(shape) == 1:
                    rows += int(shape[0])
            elif entry.filename.endswith('_labeled_mask.npy'):
                masks += 1
    return {'exists': True, 'client_index_arrays': clients, 'label_mask_arrays': masks,
            'index_rows_from_headers': rows,
            'index_space': 'post-train-validation-subset; not original benchmark IDs',
            'content_read': False}


def inspect_run(config_path, inspect_headers=False):
    directory = config_path.parent
    cfg = OmegaConf.load(config_path)
    inputs = read_json(directory / 'inputs.json') or {}
    if not isinstance(inputs, dict):
        raise ValueError('inputs.json must be a JSON object')
    records = 0
    fixed_test = False
    latest_step = None
    metric_path = directory / 'metrics_by_checkpoint.jsonl'
    if metric_path.is_file():
        with metric_path.open(encoding='utf-8') as stream:
            for number, line in enumerate(stream):
                if number >= 100000 or len(line) > 1024 * 1024:
                    raise ValueError('Metrics inspection size limit exceeded')
                if not line.strip():
                    continue
                item = json.loads(line)
                records += 1
                latest_step = item.get('step')
                fixed_test = fixed_test or item.get('split') == 'fixed_horizon_test'
    artifact_candidates = sorted({str(path.relative_to(directory))
                                  for pattern in ('*diagnos*', '*split*manifest*')
                                  for path in directory.glob(pattern) if path.is_file()})
    fields = [name for name in DIAGNOSTIC_FIELDS if OmegaConf.select(cfg, name) is not None]
    source_hash = (inputs.get('source_sha256') or {}).get('datasets/SemiDatasets.py')
    current_source = PROJECT / 'datasets/SemiDatasets.py'
    source_matches = (hashlib.sha256(current_source.read_bytes()).hexdigest() == source_hash
                      if source_hash is not None and current_source.is_file() else None)
    dataset = str(OmegaConf.select(cfg, 'datasets.name', default='unknown'))
    val_ratio = float(OmegaConf.select(cfg, 'datasets.val_ratio', default=0.0))
    return {
        'run_directory': str(directory),
        'dataset': dataset,
        'variant': OmegaConf.select(cfg, 'experiment.variant'),
        'seed': OmegaConf.select(cfg, 'datasets.seed', default=0),
        'num_clients_configured': OmegaConf.select(cfg, 'topology.num_clients'),
        'registered_clients_recorded': len(inputs.get('client_edges') or {}),
        'val_ratio': val_ratio,
        'validation_uses_official_test_fallback': val_ratio <= 0,
        'input_protocol': inputs.get('input_protocol', 'not_recorded'),
        'final_checkpoint_metadata': metadata(directory / 'final.pth'),
        'checkpoint_deserialized': False,
        'metric_records': records,
        'last_recorded_step': latest_step,
        'fixed_horizon_test_record_found': fixed_test,
        'diagnostic_config_fields_present': fields,
        'diagnostic_named_artifacts': artifact_candidates,
        'dataset_source_matches_current': source_matches,
        'partition_metadata': (partition_headers(directory / 'partition_masks.npz') if inspect_headers
                               else metadata(directory / 'partition_masks.npz')),
        'e2_independence_status': 'UNCONFIRMED',
        'reason': (
            'Current E1 code has no diagnostic reservation. Named artifacts/config fields require '
            'their actual split provenance and code to be checked.' if fields or artifact_candidates else
            'No explicit diagnostic field/artifact found; current E1 only builds train/validation/test.'
        ),
        'official_test_reuse_note': (
            'CIFAR official train is allocated to train/validation; official test is evaluated at '
            'the final/best checkpoints. Do not relabel either as an independent diagnostic split.'
            if dataset in {'cifar10', 'cifar100'} else
            'HARBox raw-file/user/window provenance and preprocessing fit scope require checking. '
            'Filenames or unused users alone do not prove independent diagnostic data.'
        ),
    }


def inspect_harbox_zip(path):
    groups = {'train': {}, 'test': {}}
    skipped = 0
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or '__MACOSX' in entry.filename:
                continue
            parts = entry.filename.replace('\\', '/').split('/')
            split = ('train' if entry.filename.endswith('_train.txt') else
                     'test' if entry.filename.endswith('_test.txt') else None)
            if split is None:
                continue
            try:
                if len(parts) < 3 or parts[-1].split('_')[0] not in ACTIVITIES:
                    raise ValueError('Unrecognized filename')
                user = int(parts[-2])
            except ValueError:
                skipped += 1
                continue
            groups[split].setdefault(user, 0)
            groups[split][user] += 1
    summary = {}
    override = os.environ.get('HARBOX_TRAIN_USERS')
    for split, files_by_user in groups.items():
        users = sorted(files_by_user)
        default_count = min(100, max(1, int(.8 * len(users))))
        configured = int(override) if override else default_count
        count = min(max(configured, 1), max(len(users)-1, 1)) if users else 0
        summary[split] = {
            'matching_files': sum(files_by_user.values()),
            'users_from_filenames_only': len(users),
            'hypothetical_train_prefix_count': count,
            'hypothetical_heldout_user_count': max(0, len(users)-count),
        }
    return {
        'zip_path': str(path), 'zip_metadata_only': True,
        'sensor_records_read': False, 'unrecognized_named_files': skipped,
        'current_HARBOX_TRAIN_USERS': override,
        'current_environment_is_not_historical_run_evidence': True,
        'by_filename_split': summary,
        'official_test_files_present': bool(groups['test']),
        'test_loader_train_file_fallback_possible': not groups['test'],
        'e2_independence_status': 'UNCONFIRMED',
        'limitations': [
            'Users with no valid windows are not detectable from filenames; actual counts may be smaller.',
            'Current ZIP loader separately applies user selection after train/test filename filtering.',
            'Raw session overlap and train-only scaler fitting are not established by ZIP metadata.',
            'Discarded files/users are investigation candidates, not automatically valid diagnostic data.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', nargs='+', default=['runs'], help='Existing run roots; read only.')
    parser.add_argument('--harbox-zip', type=Path, help='Optional explicit ZIP; lists directory metadata only.')
    parser.add_argument('--inspect-partition-headers', action='store_true')
    parser.add_argument('--max-runs', type=int, default=100)
    args = parser.parse_args()
    if args.max_runs < 1:
        parser.error('--max-runs must be positive')
    reports, errors, missing = [], [], []
    seen = set()
    truncated = False
    for raw_root in args.runs_root:
        root = Path(raw_root)
        if not root.is_dir():
            missing.append(str(root))
            continue
        for path in root.rglob('config_resolved.yaml'):
            identity = str(path.resolve())
            if identity in seen:
                continue
            if len(seen) >= args.max_runs:
                truncated = True
                break
            seen.add(identity)
            try:
                reports.append(inspect_run(path, args.inspect_partition_headers))
            except Exception as error:
                errors.append({'run_directory': str(path.parent), 'error_type': type(error).__name__,
                               'message': 'Metadata could not be safely inspected; check this run locally.'})
    archive = None
    if args.harbox_zip is not None:
        try:
            archive = inspect_harbox_zip(args.harbox_zip)
        except Exception as error:
            errors.append({'zip_path': str(args.harbox_zip), 'error_type': type(error).__name__,
                           'message': 'ZIP metadata inspection failed; no entries extracted.'})
    result = {
        'audit_type': 'e2_read_only_inventory_v1',
        'scope': 'No models/data arrays loaded, no training, downloads, online trackers or files written.',
        'runs': reports, 'missing_roots': missing, 'truncated': truncated,
        'harbox_zip': archive, 'errors': errors,
        'conclusion': 'No automatic proof of independent diagnostic data. Inspect provenance before E2.',
    }
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())

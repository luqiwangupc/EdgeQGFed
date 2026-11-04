"""Local E1 records. No dataset/model loading and no remote logging side effects."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


class E1RunLogger:
    def __init__(self, config, aggregator):
        self.stress_source = OmegaConf.select(config, 'stress.source_edge') if bool(
            OmegaConf.select(config, 'stress.enabled', default=False)) else None
        root = Path(OmegaConf.select(config, 'experiment.output_dir', default='runs/e1'))
        root.mkdir(parents=True, exist_ok=True)
        variant = OmegaConf.select(config, 'experiment.variant', default=aggregator.mode)
        seed = int(OmegaConf.select(config, 'datasets.seed', default=0))
        self.directory = Path(tempfile.mkdtemp(prefix=f'{variant}_s{seed}_', dir=str(root)))
        self.teacher_version = 0
        self.last_evaluation = None
        # Each invocation gets its own checkpoint directory, including repeated same-seed runs.
        config.train.ckpt_save_path = str(self.directory / 'checkpoints')
        OmegaConf.save(config, self.directory / 'config_resolved.yaml', resolve=True)

    def _append(self, filename, record):
        with (self.directory / filename).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

    @staticmethod
    def _model_hash(model):
        digest = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            value = tensor.detach().cpu().contiguous()
            digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode('utf-8'))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def record_inputs(self, tree, config, encoder=None):
        arrays = {}
        clients = {}
        for client_id, entry in tree.client_registry.items():
            arrays[f'{client_id}_indices'] = np.asarray(entry['indices'], dtype=np.int64)
            arrays[f'{client_id}_labeled_mask'] = entry['labeled_mask'].cpu().numpy()
            clients[client_id] = entry['edge_name']
        np.savez_compressed(self.directory / 'partition_masks.npz', **arrays)
        if tree.stress_manifest is not None:
            (self.directory / 'stress_manifest.json').write_text(
                json.dumps(tree.stress_manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        project = Path(__file__).resolve().parents[1]
        sources = [project / 'EdgeQGFed.py']
        for folder in ('datasets', 'models', 'tree', 'utils'):
            sources.extend(sorted((project / folder).glob('*.py')))
        record = {
            'seed': int(OmegaConf.select(config, 'datasets.seed', default=0)),
            'stress_mask_sha256': tree.stress_manifest['mask_sha256'] if tree.stress_manifest is not None else None,
            'client_edges': clients,
            'client_network_profiles': tree.client_network_profile,
            'edge_network_profiles': tree.edge_network_profile,
            'initial_teacher_sha256': self._model_hash(tree.root.model.model),
            'initial_edge_sha256': {edge.name: self._model_hash(edge.model) for edge in tree.root.children},
            'encoder_sha256': self._model_hash(encoder) if encoder is not None else None,
            'source_sha256': {path.relative_to(project).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sources},
            'summary_window': 'current_sync_batch',
            'rho_scope': 'all_unlabeled_including_zero_weights',
            'confidence_scope': 'teacher_unlabeled',
            'paired_inputs': bool(OmegaConf.select(config, 'experiment.paired_inputs', default=False)),
            'input_protocol': ('client_keyed_v2' if tree.client_batch_loader is not None else 'edge_step_v1'),
            'client_batch_workers': OmegaConf.select(config, 'datasets.client_batch_workers'),
            'cpu_threads': OmegaConf.select(config, 'train.cpu_threads'),
            'selection_protocol': ('available_feasible_v1' if OmegaConf.select(config, 'network.budget.strict', default=False)
                                   else 'legacy_selected_then_drop'),
            'communication_protocol': OmegaConf.select(config, 'network.budget.protocol', default='legacy_partial_mib'),
            'loss_flags': OmegaConf.to_container(config.train.losses) if OmegaConf.select(config, 'train.losses') is not None else None,
            'harbox_environment': {name: os.environ.get(name) for name in
                                   ('HARBOX_ROOT', 'HARBOX_TRAIN_USERS', 'HARBOX_WINDOW_SIZE', 'HARBOX_WINDOW_STRIDE')},
            'random_stream_scheme': (
                'client_keyed_v2: availability/batch/views by seed/client/local_step; model by edge_step'
                if tree.client_batch_loader is not None else
                'edgeqgfed-e1-v1: seed/phase/local_step (see utils/e1_random.py)'
            ),
            'pairing_limit': 'Replay uses keyed RNGs and saved configuration, not serialized views; verify hashes/traces. GPU kernels are not forced deterministic.',
        }
        (self.directory / 'inputs.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8',
        )

    def record_participation(self, step, sampled, network_state):
        record = {
            'local_step': step,
            'selected': sampled,
            'successful': network_state['active_client_ids_by_edge'],
        }
        if 'client_batch_plan' in network_state:
            record['client_batch_plan'] = network_state['client_batch_plan']
        if 'selection' in network_state:
            record['selection'] = network_state['selection']
        self._append('participation.jsonl', record)

    def record_communication(self, step, client_mb, summary_mb, model_mb, teacher_mb,
                             cumulative_mb, budget_mb, selection):
        parts = dict(client_to_edge=round(client_mb*1_000_000),
                     edge_summary_to_cloud=round(summary_mb*1_000_000),
                     model_to_all_edges=round(model_mb*1_000_000),
                     teacher_to_all_edges=round(teacher_mb*1_000_000))
        self._append('communication_by_round.jsonl', {
            'local_step': step, 'protocol': 'matched_tensor_payload_v1', 'unit': 'bytes',
            'by_direction': parts, 'round_bytes': sum(parts.values()),
            'cumulative_bytes': round(cumulative_mb*1_000_000),
            'budget_bytes': round(budget_mb*1_000_000),
            'reserved_payload_bytes': selection['reserved_payload_bytes'],
            'scope': 'modeled tensor payload only; not measured links or complete E6 compute/energy',
        })

    def record_evaluation(self, step, cloud, edges, costs, split='validation'):
        record = {'step': step, 'teacher_version': self.teacher_version, 'split': split,
                  'cloud': cloud, 'shared_split_edge_mean': edges, 'costs': costs}
        self._append('metrics_by_checkpoint.jsonl', record)
        if split == 'validation':
            self.last_evaluation = record

    def record_aggregation(self, step, result, summaries, ema_algorithm, ema_decay, downlink_edges):
        mu = float(ema_decay) if ema_algorithm == 'AEMA' else 0.0
        q = result['source_weights']
        version_before = self.teacher_version
        if q is not None:
            self.teacher_version += 1
        self._append('aggregation_events.jsonl', {
            'local_step': step,
            'teacher_version_before': version_before,
            'teacher_version_after': self.teacher_version,
            'mode': result['mode'],
            'teacher_weight_mode': result['teacher_weight_mode'],
            'active_edge_ids': result['active_edge_ids'],
            'A_recipient_by_source': result['attention_matrix'].tolist() if q is not None else None,
            'omega': result['teacher_weights'].tolist() if q is not None else None,
            'q': q.tolist() if q is not None else None,
            'mu_used': mu,
            'current_teacher_source_mass': ((1.0 - mu) * q).tolist() if q is not None else None,
            'designated_stress_source': self.stress_source,
            'stress_source_active': self.stress_source in result['active_edge_ids'] if self.stress_source else None,
            'stress_source_q': (float(q[result['active_edge_ids'].index(self.stress_source)])
                                if self.stress_source in result['active_edge_ids'] and q is not None else 0.0)
                               if self.stress_source else None,
            'stress_source_current_teacher_mass': ((1.0-mu)*float(q[result['active_edge_ids'].index(self.stress_source)])
                                                  if self.stress_source in result['active_edge_ids'] and q is not None else 0.0)
                                                 if self.stress_source else None,
            'summary_window': 'current_sync_batch',
            'summaries': [{
                'edge_id': s['name'], 'N': s['batch_size'], 'rho': s['pseudo_quality'],
                'r': s['labeled_ratio'], 'c': s['mean_confidence'],
                'prototype_counts': s['prototype_counts'].detach().cpu().tolist(),
                'quality_observed': s.get('quality_observed', True),
            } for s in summaries],
            'dispatch': {name: ('personalized' if name in result['active_edge_ids']
                                and result['mode'].endswith('_graph') else 'global_aggregate')
                         for name in downlink_edges} if q is not None else {},
        })

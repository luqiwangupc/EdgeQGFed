"""Opt-in E3 training-label stress and observational validation mix probes."""

import hashlib
import json
import math

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from utils.e1_random import e1_random_stream


def validate_stress_config(config):
    if not bool(OmegaConf.select(config, 'stress.enabled', default=False)):
        return False
    rate = float(config.stress.label_corruption_rate)
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError('stress.label_corruption_rate must be in [0, 1]')
    source = str(config.stress.source_edge)
    if source not in {f'Edge{i}' for i in range(int(config.topology.num_edges))}:
        raise ValueError(f'Unknown E3 source ES: {source}')
    if str(OmegaConf.select(config, 'network.mobility.mode', default='static')) != 'static':
        raise ValueError('First E3 protocol requires static client/ES association')
    if float(config.models.attack_rate) != 0 or bool(config.train.differential.use):
        raise ValueError('E3 label stress must not be combined with legacy attack_rate or DP')
    if str(OmegaConf.select(config, 'train.class_balance.mode', default='none')).lower() not in {'none', 'off', 'false'}:
        raise ValueError('First E3 protocol requires unweighted classification loss')
    if float(OmegaConf.select(config, 'datasets.balanced_labeled_batch_fraction', default=0)) != 0:
        raise ValueError('First E3 protocol requires label-independent batch sampling')
    if (not bool(OmegaConf.select(config, 'experiment.e1_logging', default=False)) or
            not bool(OmegaConf.select(config, 'experiment.paired_inputs', default=False)) or
            OmegaConf.select(config, 'datasets.client_batch_workers') is None):
        raise ValueError('E3 requires local records and the paired client-keyed input backend')
    if not 0 < float(OmegaConf.select(config, 'datasets.val_ratio', default=0)) < 1:
        raise ValueError('E3 requires held-out validation; never use final test for probes')
    if int(config.stress.probe_samples) < 1 or int(config.stress.probe_every_steps) < 1:
        raise ValueError('E3 probe size and interval must be positive')
    return True


def apply_label_stress(client_registry, config, num_classes):
    """Overrides are local positions; clean targets/masks/indices remain immutable."""
    if not validate_stress_config(config):
        return None
    source = str(config.stress.source_edge)
    pool = []
    for client_id, entry in sorted(client_registry.items()):
        if entry['edge_name'] != source:
            continue
        for pos in range(len(entry['indices'])):
            if bool(entry['labeled_mask'][pos]):
                index = int(entry['indices'][pos])
                pool.append([client_id, pos, index, int(entry['targets'][index])])
    if not pool or num_classes < 2:
        raise ValueError('E3 source needs labeled training data and at least two classes')
    if any(not 0 <= row[3] < num_classes for row in pool):
        raise ValueError('E3 source labels outside class range')
    if any(entry.get('training_label_overrides') for entry in client_registry.values()):
        raise ValueError('E3 training label overrides already installed')
    seed = int(config.datasets.seed)
    mask_seed = int(config.stress.mask_seed)
    key = f'edgeqgfed-e3-labels-v1:{seed}:{mask_seed}:{source}'
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))
    # A common permutation makes lower-rate masks nested in higher-rate masks.
    order = rng.permutation(len(pool))
    replacements = [(label + int(rng.integers(1, num_classes))) % num_classes
                    for _, _, _, label in pool]
    count = int(math.floor(len(pool) * float(config.stress.label_corruption_rate)))
    changes = []
    for chosen in sorted(order[:count].tolist()):
        client_id, pos, index, old = pool[chosen]
        new = replacements[chosen]
        client_registry[client_id].setdefault('training_label_overrides', {})[pos] = new
        changes.append(dict(client_id=client_id, local_position=pos,
                            train_subset_index=index, original_label=old, training_label=new))
    encode = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return dict(protocol='fixed_source_training_label_v1', source_edge=source, seed=seed,
                mask_seed=mask_seed, requested_rate=float(config.stress.label_corruption_rate),
                labeled_source_samples=len(pool), corrupted_samples=count, actual_rate=count/len(pool),
                index_space='entry.indices after train/validation subsetting; not raw benchmark index',
                labeled_pool_sha256=hashlib.sha256(encode(pool)).hexdigest(),
                mask_sha256=hashlib.sha256(encode(changes)).hexdigest(), changes=changes)


class ValidationMixProbe:
    """Fixed validation subset, unweighted CE; no gradient or aggregation feedback.

    The pre-mix model is a single-step no-mix counterfactual on the SAME history,
    not an end-to-end local-only reference. Cached frozen features stay on CPU.
    """

    def __init__(self, config, valset, encoder, device):
        self.source = str(config.stress.source_edge)
        self.interval = int(config.stress.probe_every_steps)
        self.final_step = int(config.train.total_steps)
        self.device = device
        self.batch_size = int(config.datasets.batch_size)
        seed = int(config.datasets.seed)
        key = hashlib.sha256(f'e3-validation-subset:{seed}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], 'little'))
        indices = np.sort(rng.choice(len(valset), min(len(valset), int(config.stress.probe_samples)), replace=False))
        if len(indices) == 0:
            raise ValueError('Empty E3 validation probe')
        self.manifest = dict(split='validation (also used for model selection; not E2 independent diagnostic)',
                             validation_positions=indices.tolist(), sample_count=len(indices),
                             indices_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                             scope='public shared validation subset; not client/user local-test',
                             reference='one-step no-mix on shared history; not local-only training')
        states = [(module, module.training) for module in encoder.modules()]
        features, labels = [], []
        try:
            encoder.eval()
            with e1_random_stream(seed, 'e3_probe_encoding', 0, device), torch.no_grad():
                loader = DataLoader(Subset(valset, indices.tolist()), batch_size=self.batch_size,
                                    shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(seed))
                for batch in loader:
                    features.append(encoder(batch['img'].to(device)).detach().cpu())
                    labels.append(batch['label'].long().cpu())
        finally:
            for module, training in states:
                module.training = training
        self.features = torch.cat(features)
        self.labels = torch.cat(labels)

    def due(self, step):
        return (step + 1) % self.interval == 0 or step + 1 == self.final_step

    @torch.no_grad()
    def evaluate(self, tree):
        results = {}
        for edge in tree.root.children:
            states = [(module, module.training) for module in edge.model.modules()]
            loss, correct = 0., 0
            try:
                edge.model.eval()
                for start in range(0, len(self.labels), self.batch_size):
                    labels = self.labels[start:start+self.batch_size].to(self.device)
                    logits = edge.model(self.features[start:start+self.batch_size].to(self.device))
                    loss += F.cross_entropy(logits, labels, reduction='sum').item()
                    correct += int((logits.argmax(1) == labels).sum().item())
            finally:
                for module, training in states:
                    module.training = training
            results[edge.name] = dict(loss=loss/len(self.labels), accuracy=100*correct/len(self.labels))
        return results

    def record(self, logger, step, before, after, result):
        active = result['active_edge_ids']
        source_pos = active.index(self.source) if self.source in active else None
        rows = []
        for name in before:
            delta = after[name]['loss'] - before[name]['loss']
            mass = 0.
            if source_pos is not None:
                mass = float(result['attention_matrix'][active.index(name), source_pos]) if name in active else float(result['source_weights'][source_pos])
            rows.append(dict(edge_id=name, clean_receiver=name != self.source, active=name in active,
                             before=before[name], after=after[name], delta_loss=delta,
                             loss_increased=delta > 1e-6, designated_source_parameter_mass=mass))
        clean = [row for row in rows if row['clean_receiver']]
        active_clean = [row for row in clean if row['active']]
        logger._append('stress_mix_probes.jsonl', dict(
            local_step=step, completed_steps=step+1, source_active=source_pos is not None,
            sample_count=len(self.labels), rows=rows, loss_tolerance=1e-6,
            clean_receiver_accuracy=sum(r['after']['accuracy'] for r in clean)/len(clean) if clean else None,
            loss_increase_fraction=sum(r['loss_increased'] for r in clean)/len(clean) if clean else None,
            active_clean_loss_increase_fraction=sum(r['loss_increased'] for r in active_clean)/len(active_clean) if active_clean else None,
            reference='one-step no-mix; includes recipient fallback dispatch, excludes teacher prediction effects',
        ))

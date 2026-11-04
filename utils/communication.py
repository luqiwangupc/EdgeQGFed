"""Units and tensor-only matched message envelope for strict E4 controls."""

from omegaconf import OmegaConf


def strict_budget_enabled(config):
    return config is not None and bool(OmegaConf.select(config, 'network.budget.strict', default=False))


def communication_unit_bytes(config=None):
    # Legacy runs retain MiB arithmetic. E4 explicitly uses decimal MB / Mbps.
    return 1_000_000 if strict_budget_enabled(config) else 1024 ** 2


def summary_metadata_bytes(num_classes, feature_dim):
    # float32 prototypes/counts, float32 rho/r/c, int64 processed N.
    return int(num_classes) * (int(feature_dim) + 1) * 4 + 3 * 4 + 8

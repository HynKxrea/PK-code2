import json
import os
from .scorer import WeightedSumScorer, GeometricScorer


def create_scorer(config_path=None):
    """
    Load scoring configuration JSON and return (scorer, direction).
    direction: 'maximize' or 'minimize'

    method 옵션:
      'weighted_sum'  — 기존 raw Q 면적 가중합
      'geometric'     — Q_eff 기반 가중합 + 파편화 패널티 (Level1+2)
    """

    default_weights = [10, 8, 6, 4, 2]
    default_direction = 'minimize'

    if config_path is None or not os.path.exists(config_path):
        return WeightedSumScorer(default_weights), default_direction

    try:
        with open(config_path, 'r', encoding='utf-8') as fh:
            cfg = json.load(fh)
    except Exception:
        return WeightedSumScorer(default_weights), default_direction

    method = cfg.get('method', 'weighted_sum')
    direction = cfg.get('direction', 'minimize')

    if method == 'weighted_sum':
        wcfg = cfg.get('weighted_sum', {})
        weights = wcfg.get('weights', default_weights)
        normalize = bool(wcfg.get('normalize', False))
        return WeightedSumScorer(weights, normalize=normalize), direction

    if method == 'geometric':
        gcfg = cfg.get('geometric', {})
        weights = gcfg.get('weights', [10, 8, 6, 4])
        frag_weight = float(gcfg.get('fragmentation_weight', 0.3))
        normalize = bool(gcfg.get('normalize', False))
        return GeometricScorer(weights, fragmentation_weight=frag_weight, normalize=normalize), direction

    # fallback
    return WeightedSumScorer(default_weights), default_direction

import json
import os
from .scorer import WeightedSumScorer, GeometricScorer


def create_scorer(config_path=None):
    """
    Load scoring configuration JSON and return (scorer, direction, q_mode).

    q_mode: 'raw' | 'effective'
    method: 'weighted_sum' | 'geometric'
    """

    default_weights_ws  = [10, 8, 6, 4]
    default_weights_geo = [10, 8, 6, 4]
    default_direction = 'minimize'
    default_q_mode = 'raw'

    if config_path is None or not os.path.exists(config_path):
        return WeightedSumScorer(default_weights_ws), default_direction, default_q_mode

    try:
        with open(config_path, 'r', encoding='utf-8') as fh:
            cfg = json.load(fh)
    except Exception:
        return WeightedSumScorer(default_weights_ws), default_direction, default_q_mode

    method    = cfg.get('method', 'weighted_sum')
    direction = cfg.get('direction', 'minimize')
    q_mode    = cfg.get('q_mode', 'raw')

    if method == 'weighted_sum':
        wcfg = cfg.get('weighted_sum', {})
        weights = wcfg.get('weights', default_weights_ws)
        normalize = bool(wcfg.get('normalize', False))
        return WeightedSumScorer(weights, normalize=normalize), direction, q_mode

    if method == 'geometric':
        gcfg = cfg.get('geometric', {})
        weights = gcfg.get('weights', default_weights_geo)
        frag_weight = float(gcfg.get('fragmentation_weight', 0.3))
        normalize = bool(gcfg.get('normalize', False))
        return GeometricScorer(weights, fragmentation_weight=frag_weight, normalize=normalize), direction, q_mode

    return WeightedSumScorer(default_weights_ws), default_direction, default_q_mode

import json
import os
from .scorer import WeightedSumScorer


def create_scorer(config_path=None):
    """
    Load scoring configuration JSON and return (scorer, direction)
    direction: 'maximize' or 'minimize' indicating whether higher scores are better.

    If config_path is None or file missing/invalid, returns a default WeightedSumScorer with weights [10,7,5,2,1]
    and direction 'minimize' (to preserve previous behavior where lower objective was preferred).
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

    # fallback
    return WeightedSumScorer(default_weights), default_direction

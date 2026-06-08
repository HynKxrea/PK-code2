from abc import ABC, abstractmethod
import numpy as np

class Scorer(ABC):
    @abstractmethod
    def score_leather(self, quality_areas):
        """
        quality_areas: sequence or array length=5 (Q1..Q5 areas, typically in mm^2)
        returns: scalar score (float). Higher means better by convention in the config (direction controls objective).
        """
        pass


class WeightedSumScorer(Scorer):
    def __init__(self, weights, normalize=False):
        self.weights = np.array(weights, dtype=float)
        if self.weights.shape[0] != 5:
            raise ValueError("weights must be length 5 for Q1..Q5")
        self.normalize = bool(normalize)

    def score_leather(self, quality_areas):
        qa = np.array(quality_areas, dtype=float)
        if qa.shape[0] < 5:
            # pad with zeros if necessary
            qa = np.pad(qa, (0, 5 - qa.shape[0]))
        s = float(np.dot(self.weights, qa))
        if self.normalize:
            total = qa.sum()
            if total > 0:
                s = s / total
        return s
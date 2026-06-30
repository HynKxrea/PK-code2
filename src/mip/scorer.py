from abc import ABC, abstractmethod
import numpy as np

class Scorer(ABC):
    @abstractmethod
    def score_leather(self, quality_areas, geo_info=None):
        """
        quality_areas: sequence or array length=5 (Q1..Q5 areas, typically in mm^2)
        geo_info: dict from defect_analyzer {"effective": [...], "largest_blob_ratio": [...]}
                  None이면 raw quality_areas만 사용
        returns: scalar score (float).
        """
        pass


class WeightedSumScorer(Scorer):
    def __init__(self, weights, normalize=False):
        self.weights = np.array(weights, dtype=float)
        if self.weights.shape[0] not in (4, 5):
            raise ValueError("weights must be length 4 or 5 for Q1..Q4(Q5)")
        self.normalize = bool(normalize)

    def score_leather(self, quality_areas, geo_info=None):
        n = len(self.weights)
        qa = np.array(quality_areas, dtype=float)
        if qa.shape[0] < n:
            qa = np.pad(qa, (0, n - qa.shape[0]))
        s = float(np.dot(self.weights, qa[:n]))
        if self.normalize:
            total = qa[:n].sum()
            if total > 0:
                s = s / total
        return s


class GeometricScorer(Scorer):
    """
    Level1: Q_eff 기반 가중합 (결함 팽창 후 실제 사용 가능 면적)
    Level2: 파편화 패널티 (최대 연속 블롭 비율로 점수 조정)

    score = Σ w_k * eff_increment_k
            * [(1 - frag_weight) + frag_weight * weighted_blob_ratio]

    geo_info 없으면 raw quality_areas의 WeightedSumScorer와 동일하게 동작.
    """

    def __init__(self, weights, fragmentation_weight=0.3, normalize=False):
        """
        weights: Q1..Q4 유효 면적 증분에 대한 가중치 (길이 4 또는 5)
        fragmentation_weight: 파편화 패널티 강도 (0=무시, 1=완전 반영)
        """
        self.weights = np.array(weights, dtype=float)[:4]
        self.fragmentation_weight = float(fragmentation_weight)
        self.normalize = bool(normalize)

    def score_leather(self, quality_areas, geo_info=None):
        qa = np.array(quality_areas, dtype=float)
        if qa.shape[0] < 4:
            qa = np.pad(qa, (0, 4 - qa.shape[0]))

        # Q는 항상 raw 면적 기반 가중합
        base = float(np.dot(self.weights, qa[:4]))

        if geo_info is None:
            # 파편화 정보 없으면 페널티 없이 raw 가중합만
            return base

        # 파편화 패널티만 geo_info에서 계산 (Q에는 raw 면적 유지)
        ratios   = geo_info["largest_blob_ratio"]  # [r1, r2, r3, r4]
        effective = geo_info["effective"]           # 면적 가중치 계산용

        eff_inc = [
            effective[0],
            effective[1] - effective[0],
            effective[2] - effective[1],
            effective[3] - effective[2],
        ]
        total_eff = sum(eff_inc) + 1e-9
        frag_ratio = sum(r * e for r, e in zip(ratios, eff_inc)) / total_eff
        penalty_factor = (1.0 - self.fragmentation_weight) + self.fragmentation_weight * frag_ratio

        score = base * penalty_factor

        if self.normalize:
            total_raw = qa[:4].sum()
            if total_raw > 1e-6:
                score /= total_raw

        return score
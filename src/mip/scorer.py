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
        if self.weights.shape[0] != 5:
            raise ValueError("weights must be length 5 for Q1..Q5")
        self.normalize = bool(normalize)

    def score_leather(self, quality_areas, geo_info=None):
        qa = np.array(quality_areas, dtype=float)
        if qa.shape[0] < 5:
            qa = np.pad(qa, (0, 5 - qa.shape[0]))
        s = float(np.dot(self.weights, qa))
        if self.normalize:
            total = qa.sum()
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
        if geo_info is None:
            # fallback: raw 면적 가중합
            qa = np.array(quality_areas, dtype=float)
            if qa.shape[0] < 4:
                qa = np.pad(qa, (0, 4 - qa.shape[0]))
            return float(np.dot(self.weights, qa[:4]))

        effective = geo_info["effective"]          # [Q1_eff, Q1Q2_eff, Q1Q2Q3_eff, Q1Q2Q3Q4_eff]
        ratios    = geo_info["largest_blob_ratio"] # [r1, r2, r3, r4]

        # 품질 레벨별 증분 유효 면적
        eff_inc = [
            effective[0],
            effective[1] - effective[0],
            effective[2] - effective[1],
            effective[3] - effective[2],
        ]

        # 유효 면적 가중합 (Level 1)
        base = float(np.dot(self.weights, eff_inc))

        # 파편화 패널티 (Level 2): 면적 가중 평균 blob ratio
        # blob_ratio=1(덩어리 하나) → factor=1.0 (패널티 없음)
        # blob_ratio=0(완전 조각) → factor=1-frag_weight (점수 낮아짐 = minimize에서 먼저 소모)
        total_eff = sum(eff_inc) + 1e-9
        frag_ratio = sum(r * e for r, e in zip(ratios, eff_inc)) / total_eff
        penalty_factor = (1.0 - self.fragmentation_weight) + self.fragmentation_weight * frag_ratio

        score = base * penalty_factor

        if self.normalize and total_eff > 1e-6:
            score /= total_eff

        return score
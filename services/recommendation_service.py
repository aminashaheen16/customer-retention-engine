"""
Retention Recommendation Engine — Step 4
Returns actionable marketing strategies based on cluster + churn risk.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetentionRecommendation:
    cluster_id:     int
    cluster_label:  str
    churn_risk:     str          # "Low" | "Medium" | "High" | "Critical"
    churn_prob:     float
    actions:        List[str]    = field(default_factory=list)
    priority:       str          = "Normal"
    estimated_impact: str        = ""

    def to_dict(self) -> dict:
        return {
            "cluster_id":       self.cluster_id,
            "cluster_label":    self.cluster_label,
            "churn_risk":       self.churn_risk,
            "churn_probability": self.churn_prob,
            "recommended_actions": self.actions,
            "priority":         self.priority,
            "estimated_impact": self.estimated_impact,
        }


# ── risk thresholds ────────────────────────────────────────────────────────
def _classify_risk(churn_prob: float) -> str:
    if churn_prob >= 0.75:
        return "Critical"
    elif churn_prob >= 0.50:
        return "High"
    elif churn_prob >= 0.25:
        return "Medium"
    return "Low"


# ── rule book ──────────────────────────────────────────────────────────────
_RULES: dict = {
    # cluster_id → risk_level → (actions, priority, impact)
    0: {  # Loyal Customers
        "Low":      (["Send loyalty reward email",
                      "Enroll in VIP program",
                      "Offer anniversary discount"],
                     "Low", "Maintain CLV — low spend needed"),
        "Medium":   (["Proactive check-in call",
                      "Exclusive loyalty bonus offer",
                      "Upgrade recommendation"],
                     "Medium", "Prevent early disengagement"),
        "High":     (["Personal account manager outreach",
                      "Retention discount (10–15%)",
                      "Service quality review"],
                     "High", "High ROI — these are long-tenure customers"),
        "Critical": (["Emergency retention call within 24h",
                      "Offer free month or service upgrade",
                      "Escalate to senior retention team"],
                     "Critical", "Losing loyal customers is very costly"),
    },
    1: {  # At-Risk Customers
        "Low":      (["Send value-reminder email",
                      "Highlight unused features"],
                     "Medium", "Early intervention opportunity"),
        "Medium":   (["Offer 3-month discount",
                      "Send retention survey",
                      "Provide contract upgrade option"],
                     "High", "Strong ROI — act before they decide to leave"),
        "High":     (["Immediate discount offer (20%)",
                      "Dedicated support contact",
                      "Contract lock-in incentive"],
                     "Critical", "High likelihood of churn without action"),
        "Critical": (["Same-day outreach call",
                      "Emergency retention package",
                      "Free premium tier for 3 months",
                      "Escalate to retention specialist"],
                     "Critical", "Imminent churn — maximum urgency"),
    },
    2: {  # New Customers
        "Low":      (["Send onboarding tutorial series",
                      "Welcome discount on next bill",
                      "Product education campaign"],
                     "Low", "Nurture early relationship"),
        "Medium":   (["Personalized onboarding call",
                      "Early-bird loyalty points",
                      "Usage tips push notifications"],
                     "Medium", "Prevent early drop-off"),
        "High":     (["Assign dedicated onboarding specialist",
                      "Offer first-month free extension",
                      "30-day satisfaction guarantee reminder"],
                     "High", "Early churn is expensive — act fast"),
        "Critical": (["Immediate satisfaction survey + call",
                      "Full first-month refund offer",
                      "Product fit assessment session"],
                     "Critical", "New customer at critical risk"),
    },
    3: {  # High-Value Customers
        "Low":      (["Offer premium tier upgrade",
                      "Exclusive beta feature access",
                      "VIP event invitation"],
                     "Low", "Maximize CLV of best segment"),
        "Medium":   (["Personal account review meeting",
                      "Tailored bundle offer",
                      "Priority support enrollment"],
                     "High", "Protect highest-value segment"),
        "High":     (["Executive outreach",
                      "Custom retention package",
                      "Revenue-match competitor pricing"],
                     "Critical", "Extremely high revenue impact if churned"),
        "Critical": (["C-level personal outreach",
                      "Custom SLA agreement",
                      "Lifetime discount negotiation",
                      "Dedicated success manager"],
                     "Critical", "Top priority — maximum retention investment justified"),
    },
}


class RecommendationService:
    """
    Rule-based retention recommendation engine.
    Input  : cluster_id + churn_probability
    Output : RetentionRecommendation
    """

    def get_recommendation(self, cluster_id: int,
                            churn_prob: float,
                            cluster_label: str = None) -> RetentionRecommendation:
        """
        Returns a RetentionRecommendation for a customer.

        Args:
            cluster_id:    KMeans cluster id (0-3)
            churn_prob:    Predicted churn probability (0.0 - 1.0)
            cluster_label: Human-readable label (optional; auto-looked up if None)

        Returns:
            RetentionRecommendation dataclass instance
        """
        from services.clustering_service import ClusteringService
        if cluster_label is None:
            cluster_label = ClusteringService.get_label(cluster_id)

        risk = _classify_risk(churn_prob)

        rule = _RULES.get(cluster_id, {}).get(risk)
        if rule:
            actions, priority, impact = rule
        else:
            actions  = ["Conduct manual customer review"]
            priority = "Medium"
            impact   = "Unknown segment — manual review needed"

        rec = RetentionRecommendation(
            cluster_id=cluster_id,
            cluster_label=cluster_label,
            churn_risk=risk,
            churn_prob=round(churn_prob, 4),
            actions=actions,
            priority=priority,
            estimated_impact=impact,
        )
        logger.info(
            "Recommendation | Cluster=%s | Risk=%s | Priority=%s",
            cluster_label, risk, priority,
        )
        return rec

    def batch_recommend(self, df: "pd.DataFrame") -> list:
        """
        Runs recommendations for a DataFrame with columns:
        [cluster_id, cluster_label, churn_probability].
        """
        results = []
        for _, row in df.iterrows():
            rec = self.get_recommendation(
                cluster_id=int(row["cluster_id"]),
                churn_prob=float(row["churn_probability"]),
                cluster_label=row.get("cluster_label"),
            )
            results.append(rec.to_dict())
        return results


if __name__ == "__main__":
    svc = RecommendationService()
    test_cases = [
        (0, 0.12),   # Loyal, Low risk
        (1, 0.78),   # At-Risk, Critical
        (2, 0.45),   # New, High
        (3, 0.30),   # High-Value, Medium
    ]
    for cluster, prob in test_cases:
        rec = svc.get_recommendation(cluster, prob)
        print(f"\n{'─'*60}")
        for k, v in rec.to_dict().items():
            print(f"  {k}: {v}")

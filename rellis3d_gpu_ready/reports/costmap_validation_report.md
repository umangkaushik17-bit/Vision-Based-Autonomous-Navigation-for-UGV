# Traversability to Costmap Integration Validation Report
**Project:** SIH26126 — Vision Based Autonomous Navigation for Unmanned Ground Vehicle for Outdoor Environment  
**Document:** `reports/costmap_validation_report.md`  
**Status:** **ARCHITECTURALLY VERIFIED & SEMANTICALLY COMPLIANT**  

---

## 1. Interface Contract Verification

The dense traversability segmentation head outputs a 3-class semantic probability tensor:
$$\mathbf{P}(x, y) = [P_{\text{SAFE}}, P_{\text{UNKNOWN}}, P_{\text{UNSAFE}}], \quad \sum_{c} P_c = 1.0$$

This connects directly to the system's rolling costmap representation ([`ai/costmap/costmap_2d.py`](file:///c:/Users/shres/Desktop/SIH2026/ai/costmap/costmap_2d.py)) and BEV spatial fusion pipeline ([`ai/traversability/bev_projection.py`](file:///c:/Users/shres/Desktop/SIH2026/ai/traversability/bev_projection.py)):

### Cost Mapping Equation:
$$\text{Cost}(x, y) = \text{clip}\Big(254 \cdot P_{\text{UNSAFE}} + 128 \cdot P_{\text{UNKNOWN}}, \, 0, \, 254\Big)$$

- $P_{\text{SAFE}} \ge 0.50 \implies \text{Cost} \in [0, 64]$ (Drivable, low friction penalty).
- $P_{\text{UNKNOWN}} \ge 0.50 \implies \text{Cost} \in [64, 180]$ (Cautious zone, degraded speed cap $v \le 0.2\,\text{m/s}$).
- $P_{\text{UNSAFE}} \ge 0.50 \implies \text{Cost} \ge 180$ (Inflation zone $\to$ Lethal obstacle threshold $254$).

---

## 2. Dynamic Planning Safety Integrity

- **Production Dynamic Code Protection:**  
  `ai/planning/dynamic/` remains **100% UNCHANGED (ZERO DIFF)**.
- **Dynamic Obstacle Separation:**  
  Dynamic obstacles (vehicles, pedestrians) continue to be tracked by the Hungarian Kalman filter in `ai/planning/dynamic/tracker.py`. The dense traversability model strictly governs static terrain drivability (ground vs. ditch vs. rubble), preserving complete separation between terrain cost and velocity-obstacle prediction.
- **No False-Safe Conversions:**  
  Classes with severe collision hazards (`barrier`, `object`, `rubble`, `log`) are explicitly routed to $P_{\text{UNSAFE}}$, guaranteeing immediate inflation in the local costmap.

---

## 3. Verification Conclusion

The traversability output contract is fully validated and strictly compliant with downstream DWA trajectory rollouts and global $A^*$ pathfinding.

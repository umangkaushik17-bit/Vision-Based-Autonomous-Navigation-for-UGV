# Autonomous UGV 100m Corridor Benchmark Report

**Execution Date:** 2026-10-05 10:04:24

## 1. Executive Performance Metrics

| Metric | Value | Units |
| :--- | :--- | :--- |
| **Mission Status** | `SUCCESS` | - |
| **Total Mission Time** | **107.80** | s |
| **Total Distance Traveled** | **106.25** | m |
| **Average Linear Speed** | 0.99 | m/s |
| **Maximum Linear Speed** | 1.02 | m/s |
| **Average Angular Speed** | 0.092 | rad/s |
| **Min Dynamic Obstacle Clearance** | **2.06** | m |
| **Dynamic Obstacles Evaluated** | 5 (W1, W2, W3, W4, W5) | units |
| **Total Collision Count** | **0** | incidents |

## 2. Waypoint Split Times

| Waypoint ID | Split Time (s) | Cumulative Dist (m) | Coordinate (x, y) |
| :--- | :--- | :--- | :--- |
| **W1** | 20.92 | 20.18 | `(19.43, -0.56)` |
| **W2** | 42.32 | 41.38 | `(39.18, 5.88)` |
| **W3** | 65.62 | 64.53 | `(59.21, -3.76)` |
| **W4** | 86.98 | 85.72 | `(79.24, 2.69)` |
| **W5** | 107.80 | 106.25 | `(99.38, 0.04)` |

## 3. Dynamic Obstacle Trajectory & Corridor Layout

- **W1 Corridor (x ≈ 16m):** `dynamic_obs_w1` - Oncoming Head-on (yaw = 180°)
- **W2 Corridor (x ≈ 32m):** `dynamic_obs_w2` - Left Lateral Cross (yaw = -90°)
- **W3 Corridor (x ≈ 55m):** `dynamic_obs_w3` - Right Lateral Cross (yaw = +90°)
- **W4 Corridor (x ≈ 73m):** `dynamic_obs_w4` - Diagonal Oncoming Intercept (yaw = -135°)
- **W5 Gateway (x ≈ 94m):** `dynamic_obs_w5` - Sentinel Blocker (yaw = 180°)

## 4. Behavioral Decision Distribution

```json
{
  "PROCEED": 410,
  "AVOID": 613,
  "WAIT": 0,
  "STOP": 0,
  "EMERGENCY_STOP": 0
}
```

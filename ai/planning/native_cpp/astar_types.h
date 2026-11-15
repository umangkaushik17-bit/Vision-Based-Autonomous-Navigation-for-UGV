#pragma once

#include <cstdint>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace astar {

struct AStarCostConfig {
    double w_distance;
    double w_traversability;
    double w_cost;
    double w_hazard;
    double w_slope;
    double w_roughness;
    double w_height;
    double w_unknown;
    double w_curvature;

    double critical_hazard_threshold; // default 0.70
    double lethal_cost_threshold;      // default 0.85
    double max_safe_slope_deg;         // default 20.0
    double max_safe_step_m;            // default 0.12
    double unknown_cost_penalty;       // default 0.80

    int32_t unknown_is_lethal;         // 1 if True, 0 if False
    int32_t heuristic_type;            // 0: Octile, 1: Euclidean
    double heuristic_weight;           // default 1.00

    int32_t enable_smoothing;          // 1 if True, 0 if False
    int32_t max_smoothing_iterations;  // default 100
    double smoothing_step_m;           // default 0.08

    int32_t enable_jps;                // 0: standard 8-connected, 1: JPS ray-skip mode
};

struct AStarGridMeta {
    double resolution_m;
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    int32_t n_rows;
    int32_t n_cols;
    int32_t is_cartesian; // 0: standard top-down (row increases as x decreases), 1: cartesian
};

struct AStarVehicleFootprint {
    double length_m;
    double width_m;
    double collision_margin_m;
    double circumscribed_radius_m;
    double inflation_radius_m;
    double max_safe_cost;              // default 0.65
};

struct AStarResult {
    int32_t status;                     // 0: SUCCESS, 1: NO_PATH, 2: INVALID_START, 3: INVALID_GOAL, 4: START_BLOCKED, 5: GOAL_BLOCKED
    int32_t raw_waypoint_count;
    int32_t smoothed_waypoint_count;
    int32_t nodes_explored;
    double path_cost;
    double path_length_m;
    double planning_latency_ms;
    double smoothing_latency_ms;
};

} // namespace astar

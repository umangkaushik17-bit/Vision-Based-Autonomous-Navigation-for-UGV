#pragma once

#include <cstdint>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dwa {

struct RobotState {
    double x;
    double y;
    double yaw;
    double v;
    double omega;
    double timestamp;
};

struct VehicleConfig {
    double length_m;
    double width_m;
    double collision_margin_m;
    double circumscribed_radius_m;
    double max_linear_velocity_mps;
    double min_linear_velocity_mps;
    double max_angular_velocity_radps;
    double max_linear_acceleration_mps2;
    double max_angular_acceleration_radps2;
    double degraded_max_linear_velocity_mps;
};

struct DWAConfig {
    double prediction_horizon_s;
    double simulation_dt_s;
    int32_t v_samples;
    int32_t omega_samples;
};

struct ScoringWeights {
    double w_path;
    double w_heading;
    double w_dist;
    double w_clear;
    double w_vel;
    double w_terrain;
};

struct DynamicConfig {
    bool enabled;
    bool enable_broad_phase;
    bool reject_high_risk;
    int32_t unknown_horizon_policy; // 0: ALLOW, 1: PENALIZE, 2: REJECT
    double unknown_horizon_penalty;
    double low_risk_penalty;
    double medium_risk_penalty;
    double high_risk_penalty;
    double w_dynamic_risk;
    double w_dynamic_clearance;
    double broad_phase_margin_m;
    double max_clearance_search_m;
};

struct CostmapMeta {
    double resolution_m;
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    int32_t n_rows;
    int32_t n_cols;
    bool is_cartesian; // true if row increases as y increases
};

struct ObstacleWaypoint {
    double x;
    double y;
    double time_offset;
    double uncertainty_radius;
};

struct ObstacleTrack {
    int32_t track_id;
    double horizon_s;
    double bbox_length;
    double bbox_width;
    double bbox_yaw;
    int32_t n_waypoints;
    const ObstacleWaypoint* waypoints;
};

struct CandidateDiagnostic {
    double v;
    double omega;
    double score;
    double static_clearance;
    double terrain_cost;
    double dyn_clearance;
    double dyn_cost;
    int32_t dyn_risk_level;
    bool is_valid;
    bool static_valid;
    bool dyn_collision;
    bool unknown_horizon;
};

struct DWAResult {
    int32_t best_idx; // -1 if no valid candidate
    double best_v;
    double best_w;
    double best_score;
    double best_static_clearance;
    double best_terrain_cost;
    double best_dyn_clearance;
    double best_dyn_cost;
    int32_t best_risk_level;

    // Diagnostics counts
    int32_t total_candidates;
    int32_t statically_valid_count;
    int32_t valid_count;
    int32_t broad_phase_cleared_count;
    int32_t expensive_collision_checks;
    int32_t dynamically_rejected_count;
    int32_t dynamically_penalized_count;
    int32_t unknown_horizon_count;

    double dwa_latency_ms;
};

} // namespace dwa

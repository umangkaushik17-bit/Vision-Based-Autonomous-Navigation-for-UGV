#include "dwa_core.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dwa {

DWAPlannerCore::DWAPlannerCore(
    const VehicleConfig& vehicle,
    const DWAConfig& dwa_cfg,
    const ScoringWeights& weights
) : vehicle_(vehicle), dwa_cfg_(dwa_cfg), weights_(weights) {
    steps_ = std::max(1, static_cast<int32_t>(std::ceil(dwa_cfg_.prediction_horizon_s / dwa_cfg_.simulation_dt_s)));
    precompute_footprint_offsets();
}

void DWAPlannerCore::precompute_footprint_offsets() {
    footprint_offsets_.clear();
    double hl = vehicle_.length_m / 2.0;
    double hw = vehicle_.width_m / 2.0;

    std::array<std::pair<double, double>, 4> corners = {{
        {hl, hw},
        {hl, -hw},
        {-hl, -hw},
        {-hl, hw}
    }};

    // 1. Corners and perimeter samples (edge_step = 0.05m)
    double edge_step = 0.05;
    for (int i = 0; i < 4; ++i) {
        double cx = corners[i].first;
        double cy = corners[i].second;
        footprint_offsets_.push_back({cx, cy});

        int next_i = (i + 1) % 4;
        double nx = corners[next_i].first;
        double ny = corners[next_i].second;
        double edge_dist = std::hypot(nx - cx, ny - cy);
        int n_sub = std::max(1, static_cast<int>(std::ceil(edge_dist / edge_step)));
        for (int k = 1; k < n_sub; ++k) {
            double alpha = static_cast<double>(k) / static_cast<double>(n_sub);
            footprint_offsets_.push_back({cx + alpha * (nx - cx), cy + alpha * (ny - cy)});
        }
    }

    // 2. Interior samples (7 x 5 grid matching CPU collision.py)
    for (int i = 0; i < 7; ++i) {
        double alpha_x = (7 > 1) ? static_cast<double>(i) / 6.0 : 0.5;
        double dx = -hl * 0.85 + alpha_x * (hl * 0.85 - (-hl * 0.85));
        for (int j = 0; j < 5; ++j) {
            double alpha_y = (5 > 1) ? static_cast<double>(j) / 4.0 : 0.5;
            double dy = -hw * 0.80 + alpha_y * (hw * 0.80 - (-hw * 0.80));
            footprint_offsets_.push_back({dx, dy});
        }
    }
}

void DWAPlannerCore::compute_dynamic_window(
    double curr_v,
    double curr_omega,
    double dt,
    bool is_degraded,
    double& v_low,
    double& v_high,
    double& w_low,
    double& w_high
) const {
    double eff_max_v = is_degraded
        ? std::min(vehicle_.max_linear_velocity_mps, vehicle_.degraded_max_linear_velocity_mps)
        : vehicle_.max_linear_velocity_mps;

    v_low = std::max(vehicle_.min_linear_velocity_mps, curr_v - vehicle_.max_linear_acceleration_mps2 * dt);
    v_high = std::min(eff_max_v, curr_v + vehicle_.max_linear_acceleration_mps2 * dt);

    w_low = std::max(-vehicle_.max_angular_velocity_radps, curr_omega - vehicle_.max_angular_acceleration_radps2 * dt);
    w_high = std::min(vehicle_.max_angular_velocity_radps, curr_omega + vehicle_.max_angular_acceleration_radps2 * dt);
}

double DWAPlannerCore::point_to_segment_dist(
    double px, double py,
    double x1, double y1,
    double x2, double y2
) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    double seg_len_sq = dx * dx + dy * dy;
    if (seg_len_sq < 1e-12) {
        return std::hypot(px - x1, py - y1);
    }
    double t = std::clamp(((px - x1) * dx + (py - y1) * dy) / seg_len_sq, 0.0, 1.0);
    double proj_x = x1 + t * dx;
    double proj_y = y1 + t * dy;
    return std::hypot(px - proj_x, py - proj_y);
}

bool DWAPlannerCore::sat_polygon_intersect(
    const std::array<std::pair<double, double>, 4>& poly1,
    const std::array<std::pair<double, double>, 4>& poly2
) {
    // 8 axes: 4 from poly1, 4 from poly2
    for (int poly_idx = 0; poly_idx < 2; ++poly_idx) {
        const auto& p = (poly_idx == 0) ? poly1 : poly2;
        for (int i = 0; i < 4; ++i) {
            int next_i = (i + 1) % 4;
            double edge_x = p[next_i].first - p[i].first;
            double edge_y = p[next_i].second - p[i].second;
            // Perpendicular axis: (-dy, dx)
            double axis_x = -edge_y;
            double axis_y = edge_x;

            // Project poly1
            double min1 = std::numeric_limits<double>::infinity();
            double max1 = -std::numeric_limits<double>::infinity();
            for (int k = 0; k < 4; ++k) {
                double proj = poly1[k].first * axis_x + poly1[k].second * axis_y;
                min1 = std::min(min1, proj);
                max1 = std::max(max1, proj);
            }

            // Project poly2
            double min2 = std::numeric_limits<double>::infinity();
            double max2 = -std::numeric_limits<double>::infinity();
            for (int k = 0; k < 4; ++k) {
                double proj = poly2[k].first * axis_x + poly2[k].second * axis_y;
                min2 = std::min(min2, proj);
                max2 = std::max(max2, proj);
            }

            // Check separation
            if (max1 < min2 - 1e-9 || max2 < min1 - 1e-9) {
                return false; // Separating axis found -> No intersection!
            }
        }
    }
    return true; // Overlap on all 8 axes -> Intersection!
}

double DWAPlannerCore::polygon_polygon_distance(
    const std::array<std::pair<double, double>, 4>& poly1,
    const std::array<std::pair<double, double>, 4>& poly2
) {
    double min_d = std::numeric_limits<double>::infinity();

    // 1. Distance from poly1 vertices to poly2 segments
    for (int i = 0; i < 4; ++i) {
        double px = poly1[i].first;
        double py = poly1[i].second;
        for (int j = 0; j < 4; ++j) {
            int next_j = (j + 1) % 4;
            double d = point_to_segment_dist(
                px, py,
                poly2[j].first, poly2[j].second,
                poly2[next_j].first, poly2[next_j].second
            );
            min_d = std::min(min_d, d);
        }
    }

    // 2. Distance from poly2 vertices to poly1 segments
    for (int i = 0; i < 4; ++i) {
        double px = poly2[i].first;
        double py = poly2[i].second;
        for (int j = 0; j < 4; ++j) {
            int next_j = (j + 1) % 4;
            double d = point_to_segment_dist(
                px, py,
                poly1[j].first, poly1[j].second,
                poly1[next_j].first, poly1[next_j].second
            );
            min_d = std::min(min_d, d);
        }
    }

    return min_d;
}

bool DWAPlannerCore::is_footprint_collision(
    double x, double y, double yaw,
    const CostmapMeta& costmap_meta,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid
) const {
    double c_yaw = std::cos(yaw);
    double s_yaw = std::sin(yaw);

    for (const auto& offset : footprint_offsets_) {
        double px = x + offset.first * c_yaw - offset.second * s_yaw;
        double py = y + offset.first * s_yaw + offset.second * c_yaw;

        // Bounds check
        if (px < costmap_meta.x_min || px >= costmap_meta.x_max ||
            py < costmap_meta.y_min || py >= costmap_meta.y_max) {
            return true;
        }

        int32_t r, c;
        if (costmap_meta.is_cartesian) {
            r = static_cast<int32_t>(std::floor((py - costmap_meta.y_min) / costmap_meta.resolution_m));
            c = static_cast<int32_t>(std::floor((px - costmap_meta.x_min) / costmap_meta.resolution_m));
        } else {
            r = static_cast<int32_t>(std::floor((costmap_meta.x_max - px) / costmap_meta.resolution_m));
            c = static_cast<int32_t>(std::floor((costmap_meta.y_max - py) / costmap_meta.resolution_m));
        }

        if (r < 0 || r >= costmap_meta.n_rows || c < 0 || c >= costmap_meta.n_cols) {
            return true;
        }

        int32_t cell_idx = r * costmap_meta.n_cols + c;
        if (obstacle_grid != nullptr && obstacle_grid[cell_idx] != 0) {
            return true;
        }
        if (hazard_grid != nullptr && hazard_grid[cell_idx] >= 0.70f) {
            return true;
        }
        if (cost_grid != nullptr && cost_grid[cell_idx] > 0.65f) {
            return true;
        }
    }
    return false;
}

bool DWAPlannerCore::interpolate_track_pose(
    const ObstacleTrack& track,
    double query_t,
    double& out_x,
    double& out_y,
    double& out_yaw,
    double& out_unc
) const {
    if (track.n_waypoints <= 0 || track.waypoints == nullptr) {
        return false;
    }
    if (query_t < -1e-6) {
        return false;
    }

    double clamped_t = std::min(query_t, track.horizon_s);
    out_yaw = track.bbox_yaw;

    // Check exact or endpoint cases
    if (clamped_t <= track.waypoints[0].time_offset) {
        out_x = track.waypoints[0].x;
        out_y = track.waypoints[0].y;
        out_unc = track.waypoints[0].uncertainty_radius;
        return true;
    }
    if (clamped_t >= track.waypoints[track.n_waypoints - 1].time_offset) {
        out_x = track.waypoints[track.n_waypoints - 1].x;
        out_y = track.waypoints[track.n_waypoints - 1].y;
        out_unc = track.waypoints[track.n_waypoints - 1].uncertainty_radius;
        return true;
    }

    // Linear interpolation between waypoints
    for (int i = 0; i < track.n_waypoints - 1; ++i) {
        const auto& w1 = track.waypoints[i];
        const auto& w2 = track.waypoints[i + 1];
        if (w1.time_offset <= clamped_t && clamped_t <= w2.time_offset) {
            double dt = w2.time_offset - w1.time_offset;
            if (dt < 1e-6) {
                out_x = w1.x;
                out_y = w1.y;
                out_unc = w1.uncertainty_radius;
                return true;
            }
            double alpha = (clamped_t - w1.time_offset) / dt;
            out_x = w1.x + alpha * (w2.x - w1.x);
            out_y = w1.y + alpha * (w2.y - w1.y);
            out_unc = w1.uncertainty_radius + alpha * (w2.uncertainty_radius - w1.uncertainty_radius);
            return true;
        }
    }
    return false;
}

void DWAPlannerCore::plan(
    const RobotState& current_state,
    const double* path_x,
    const double* path_y,
    int32_t path_len,
    double local_target_x,
    double local_target_y,
    double d_curr,
    const CostmapMeta& costmap_meta,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid,
    const uint8_t* unknown_grid,
    const float* traversability_grid,
    const float* slope_grid,
    const float* roughness_grid,
    const float* height_var_grid,
    const DynamicConfig& dyn_cfg,
    const ObstacleTrack* tracks,
    int32_t num_tracks,
    bool is_degraded,
    DWAResult& result,
    CandidateDiagnostic* candidate_diagnostics,
    double* out_trajectories
) {
    auto t0 = std::chrono::high_resolution_clock::now();

    double v_low, v_high, w_low, w_high;
    compute_dynamic_window(
        current_state.v, current_state.omega, dwa_cfg_.simulation_dt_s, is_degraded,
        v_low, v_high, w_low, w_high
    );

    int32_t n_v = dwa_cfg_.v_samples;
    int32_t n_w = dwa_cfg_.omega_samples;
    int32_t total_k = n_v * n_w;

    result.total_candidates = total_k;
    result.best_idx = -1;
    result.best_score = -std::numeric_limits<double>::infinity();
    result.best_v = 0.0;
    result.best_w = 0.0;
    result.best_static_clearance = 0.0;
    result.best_terrain_cost = 1.0;
    result.best_dyn_clearance = 0.0;
    result.best_dyn_cost = 0.0;
    result.best_risk_level = 0;

    int32_t statically_valid_count = 0;
    int32_t valid_count = 0;
    int32_t broad_phase_cleared_count = 0;
    int32_t expensive_collision_checks = 0;
    int32_t dynamically_rejected_count = 0;
    int32_t dynamically_penalized_count = 0;
    int32_t unknown_horizon_count = 0;

    double dt = dwa_cfg_.simulation_dt_s;
    double max_v = vehicle_.max_linear_velocity_mps;
    double horizon = dwa_cfg_.prediction_horizon_s;

    // Obstacle search radius
    double r_circumscribed = vehicle_.circumscribed_radius_m;
    double max_clearance_search = dyn_cfg.max_clearance_search_m > 0.0 ? dyn_cfg.max_clearance_search_m : 5.0;

    // Evaluate candidates in parallel with OpenMP
    #pragma omp parallel for schedule(dynamic) reduction(+:statically_valid_count, valid_count, \
        broad_phase_cleared_count, expensive_collision_checks, dynamically_rejected_count, \
        dynamically_penalized_count, unknown_horizon_count)
    for (int k = 0; k < total_k; ++k) {
        int i_v = k / n_w;
        int i_w = k % n_w;

        double v = (n_v > 1) ? v_low + (static_cast<double>(i_v) / (n_v - 1)) * (v_high - v_low) : v_low;
        double w = (n_w > 1) ? w_low + (static_cast<double>(i_w) / (n_w - 1)) * (w_high - w_low) : w_low;

        CandidateDiagnostic& diag = candidate_diagnostics[k];
        diag.v = v;
        diag.omega = w;
        diag.score = -std::numeric_limits<double>::infinity();
        diag.static_clearance = 5.0;
        diag.terrain_cost = 0.0;
        diag.dyn_clearance = 5.0;
        diag.dyn_cost = 0.0;
        diag.dyn_risk_level = 0;
        diag.is_valid = false;
        diag.static_valid = false;
        diag.dyn_collision = false;
        diag.unknown_horizon = false;

        // Rollout trajectory
        double curr_x = current_state.x;
        double curr_y = current_state.y;
        double curr_yaw = current_state.yaw;

        bool static_collision = false;
        double accum_terrain = 0.0;
        double min_static_clr = 5.0;

        // Buffer trajectory states for this candidate
        double traj_pts[15][3]; // [step][0: x, 1: y, 2: yaw]

        for (int s = 0; s < steps_; ++s) {
            double theta_mid = curr_yaw + 0.5 * w * dt;
            curr_x += v * std::cos(theta_mid) * dt;
            curr_y += v * std::sin(theta_mid) * dt;
            curr_yaw += w * dt;
            curr_yaw = std::remainder(curr_yaw, 2.0 * M_PI);

            traj_pts[s][0] = curr_x;
            traj_pts[s][1] = curr_y;
            traj_pts[s][2] = curr_yaw;

            if (out_trajectories != nullptr) {
                int out_idx = (k * steps_ + s) * 3;
                out_trajectories[out_idx] = curr_x;
                out_trajectories[out_idx + 1] = curr_y;
                out_trajectories[out_idx + 2] = curr_yaw;
            }

            if (static_collision) {
                continue;
            }

            // Footprint check against costmap
            if (is_footprint_collision(curr_x, curr_y, curr_yaw, costmap_meta, obstacle_grid, cost_grid, hazard_grid)) {
                static_collision = true;
            }

            // Cell terrain cost at center
            if (!static_collision && cost_grid != nullptr) {
                int32_t cr, cc;
                if (costmap_meta.is_cartesian) {
                    cr = std::clamp(static_cast<int32_t>(std::floor((curr_y - costmap_meta.y_min) / costmap_meta.resolution_m)), 0, costmap_meta.n_rows - 1);
                    cc = std::clamp(static_cast<int32_t>(std::floor((curr_x - costmap_meta.x_min) / costmap_meta.resolution_m)), 0, costmap_meta.n_cols - 1);
                } else {
                    cr = std::clamp(static_cast<int32_t>(std::floor((costmap_meta.x_max - curr_x) / costmap_meta.resolution_m)), 0, costmap_meta.n_rows - 1);
                    cc = std::clamp(static_cast<int32_t>(std::floor((costmap_meta.y_max - curr_y) / costmap_meta.resolution_m)), 0, costmap_meta.n_cols - 1);
                }

                int32_t cell_idx = cr * costmap_meta.n_cols + cc;
                double c_base = cost_grid[cell_idx];
                double c_trav = (traversability_grid != nullptr) ? traversability_grid[cell_idx] : 0.5;
                double c_haz = (hazard_grid != nullptr) ? hazard_grid[cell_idx] : 0.0;
                double c_slope = (slope_grid != nullptr) ? slope_grid[cell_idx] : 0.0;
                double c_rough = (roughness_grid != nullptr) ? roughness_grid[cell_idx] : 0.0;
                double c_hvar = (height_var_grid != nullptr) ? height_var_grid[cell_idx] : 0.0;
                double c_unk = (unknown_grid != nullptr && unknown_grid[cell_idx] != 0) ? 0.80 : 0.0;

                double step_cost = 1.50 * c_base + 0.80 * (1.0 - c_trav) + 2.50 * c_haz +
                                   1.20 * std::clamp(c_slope / 20.0, 0.0, 2.0) +
                                   0.60 * c_rough + 1.00 * std::clamp(c_hvar / 0.12, 0.0, 2.0) +
                                   1.00 * c_unk;

                if (c_base >= 0.85f || c_haz >= 0.70f || (obstacle_grid != nullptr && obstacle_grid[cell_idx] != 0)) {
                    static_collision = true;
                } else {
                    accum_terrain += step_cost;
                }
            }
        }

        if (static_collision) {
            continue;
        }

        statically_valid_count++;
        diag.static_valid = true;
        diag.terrain_cost = accum_terrain / std::max(1, steps_);

        // Dynamic obstacle evaluation
        bool dyn_rejected = false;
        double dyn_penalty = 0.0;
        double dyn_clearance = 5.0;

        if (dyn_cfg.enabled && num_tracks > 0 && tracks != nullptr) {
            double hl_ugv = (vehicle_.length_m / 2.0) + vehicle_.collision_margin_m;
            double hw_ugv = (vehicle_.width_m / 2.0) + vehicle_.collision_margin_m;

            for (int t_idx = 0; t_idx < num_tracks; ++t_idx) {
                const auto& trk = tracks[t_idx];
                double hl_obs = trk.bbox_length / 2.0;
                double hw_obs = trk.bbox_width / 2.0;
                double r_obs = std::hypot(hl_obs, hw_obs);

                for (int s = 0; s < steps_; ++s) {
                    double t_s = (s + 1) * dt;
                    if (t_s > trk.horizon_s + 1e-4) {
                        diag.unknown_horizon = true;
                        unknown_horizon_count++;
                        if (dyn_cfg.unknown_horizon_policy == 2) { // REJECT
                            dyn_rejected = true;
                            break;
                        } else if (dyn_cfg.unknown_horizon_policy == 1) { // PENALIZE
                            dyn_penalty = std::max(dyn_penalty, dyn_cfg.unknown_horizon_penalty);
                        }
                    }

                    double ox, oy, oyaw, or_unc;
                    if (!interpolate_track_pose(trk, t_s, ox, oy, oyaw, or_unc)) {
                        continue;
                    }

                    double ux = traj_pts[s][0];
                    double uy = traj_pts[s][1];
                    double uyaw = traj_pts[s][2];

                    // Broad-phase spherical distance check
                    double d_center = std::hypot(ux - ox, uy - oy);
                    double broad_thresh = r_circumscribed + r_obs + max_clearance_search;
                    if (d_center > broad_thresh) {
                        broad_phase_cleared_count++;
                        continue;
                    }

                    expensive_collision_checks++;

                    // UGV OBB corners
                    double c_u = std::cos(uyaw);
                    double s_u = std::sin(uyaw);
                    std::array<std::pair<double, double>, 4> ugv_obb = {{
                        {ux + hl_ugv * c_u - hw_ugv * s_u, uy + hl_ugv * s_u + hw_ugv * c_u},
                        {ux + hl_ugv * c_u + hw_ugv * s_u, uy + hl_ugv * s_u - hw_ugv * c_u},
                        {ux - hl_ugv * c_u + hw_ugv * s_u, uy - hl_ugv * s_u - hw_ugv * c_u},
                        {ux - hl_ugv * c_u - hw_ugv * s_u, uy - hl_ugv * s_u + hw_ugv * c_u}
                    }};

                    // Obstacle OBB corners
                    double c_o = std::cos(oyaw);
                    double s_o = std::sin(oyaw);
                    std::array<std::pair<double, double>, 4> obs_obb = {{
                        {ox + hl_obs * c_o - hw_obs * s_o, oy + hl_obs * s_o + hw_obs * c_o},
                        {ox + hl_obs * c_o + hw_obs * s_o, oy + hl_obs * s_o - hw_obs * c_o},
                        {ox - hl_obs * c_o + hw_obs * s_o, oy - hl_obs * s_o - hw_obs * c_o},
                        {ox - hl_obs * c_o - hw_obs * s_o, oy - hl_obs * s_o + hw_obs * c_o}
                    }};

                    // 8-axis SAT Intersection Check
                    if (sat_polygon_intersect(ugv_obb, obs_obb)) {
                        diag.dyn_collision = true;
                        dyn_rejected = true;
                        dynamically_rejected_count++;
                        break;
                    }

                    // Separation distance
                    double clr = polygon_polygon_distance(ugv_obb, obs_obb);
                    dyn_clearance = std::min(dyn_clearance, clr);

                    // Risk evaluation matching PredictiveCollisionChecker (lines 700-729)
                    int32_t risk_level = 0;
                    double risk_cost = 0.0;

                    if (clr <= or_unc) {
                        double sigma_pos = std::max(1e-4, or_unc / 3.0);
                        if (clr <= sigma_pos) {
                            risk_level = 3; // HIGH
                            risk_cost = 0.95 - 0.25 * (clr / sigma_pos);
                        } else if (clr <= 2.0 * sigma_pos) {
                            risk_level = 2; // MEDIUM
                            risk_cost = 0.70 - 0.30 * ((clr - sigma_pos) / sigma_pos);
                        } else {
                            risk_level = 1; // LOW
                            risk_cost = 0.40 - 0.25 * ((clr - 2.0 * sigma_pos) / sigma_pos);
                        }
                    }

                    diag.dyn_risk_level = std::max(diag.dyn_risk_level, risk_level);
                    diag.dyn_cost = std::max(diag.dyn_cost, risk_cost);

                    if (risk_level >= 3 && dyn_cfg.reject_high_risk) {
                        dyn_rejected = true;
                        dynamically_rejected_count++;
                        break;
                    } else if (risk_level >= 3) {
                        dynamically_penalized_count++;
                        dyn_penalty = std::max(dyn_penalty, dyn_cfg.high_risk_penalty + dyn_cfg.w_dynamic_risk * risk_cost);
                    } else if (risk_level == 2) {
                        dynamically_penalized_count++;
                        dyn_penalty = std::max(dyn_penalty, dyn_cfg.medium_risk_penalty + dyn_cfg.w_dynamic_risk * risk_cost);
                    } else if (risk_level == 1) {
                        dynamically_penalized_count++;
                        dyn_penalty = std::max(dyn_penalty, dyn_cfg.low_risk_penalty + 0.5 * dyn_cfg.w_dynamic_risk * risk_cost);
                    }
                }
                if (dyn_rejected) {
                    break;
                }
            }
        }

        if (dyn_rejected) {
            continue;
        }

        diag.is_valid = true;
        valid_count++;
        diag.dyn_clearance = dyn_clearance;

        // Trajectory Scoring
        double end_x = traj_pts[steps_ - 1][0];
        double end_y = traj_pts[steps_ - 1][1];
        double end_yaw = traj_pts[steps_ - 1][2];

        // 1. Path alignment score
        double path_score = 1.0;
        if (path_len >= 2 && path_x != nullptr && path_y != nullptr) {
            double min_cte = std::numeric_limits<double>::infinity();
            for (int p_idx = 0; p_idx < path_len - 1; ++p_idx) {
                double cte = point_to_segment_dist(
                    end_x, end_y,
                    path_x[p_idx], path_y[p_idx],
                    path_x[p_idx + 1], path_y[p_idx + 1]
                );
                min_cte = std::min(min_cte, cte);
            }
            path_score = 1.0 / (1.0 + min_cte);
        } else if (path_len == 1 && path_x != nullptr && path_y != nullptr) {
            double cte = std::hypot(end_x - path_x[0], end_y - path_y[0]);
            path_score = 1.0 / (1.0 + cte);
        }

        // 2. Heading score (alignment to local target direction from current state)
        double target_yaw = std::atan2(local_target_y - current_state.y, local_target_x - current_state.x);
        double heading_err = std::abs(std::remainder(target_yaw - end_yaw, 2.0 * M_PI));
        double heading_score = (M_PI - heading_err) / M_PI;

        // 3. Distance and progress score
        double d_end = std::hypot(local_target_x - end_x, local_target_y - end_y);
        double progress = (d_curr - d_end) / std::max(0.1, max_v * horizon);
        double dist_score = 0.5 * (1.0 / (1.0 + d_end)) + 0.5 * std::max(0.0, progress);

        // 4. Static clearance score
        double clearance_score = std::min(diag.static_clearance, 1.50) / 1.50;

        // 5. Velocity score (prefer forward motion aligned with path heading, penalize unnecessary reverse)
        double vel_factor = std::max(0.0, std::cos(heading_err));
        double vel_score = (v > 0.05)
            ? (v / std::max(0.1, max_v)) * vel_factor
            : (v >= 0.0 ? 0.0 : -0.5 * std::abs(v) / max_v);

        // 6. Terrain score
        double terrain_score = 1.0 / (1.0 + diag.terrain_cost);

        // 7. Dynamic reward
        double dyn_clear_reward = (dyn_cfg.enabled) ? dyn_cfg.w_dynamic_clearance * (std::min(dyn_clearance, 1.50) / 1.50) : 0.0;

        // Composite score
        double total_score = weights_.w_path * path_score +
                             weights_.w_heading * heading_score +
                             weights_.w_dist * dist_score +
                             weights_.w_clear * clearance_score +
                             weights_.w_vel * vel_score +
                             weights_.w_terrain * terrain_score -
                             dyn_penalty +
                             dyn_clear_reward;

        diag.score = total_score;
    }

    // Argmax reduction to pick best candidate
    for (int k = 0; k < total_k; ++k) {
        const auto& diag = candidate_diagnostics[k];
        if (diag.is_valid && diag.score > result.best_score) {
            result.best_score = diag.score;
            result.best_idx = k;
            result.best_v = diag.v;
            result.best_w = diag.omega;
            result.best_static_clearance = diag.static_clearance;
            result.best_terrain_cost = diag.terrain_cost;
            result.best_dyn_clearance = diag.dyn_clearance;
            result.best_dyn_cost = diag.dyn_cost;
            result.best_risk_level = diag.dyn_risk_level;
        }
    }

    result.statically_valid_count = statically_valid_count;
    result.valid_count = valid_count;
    result.broad_phase_cleared_count = broad_phase_cleared_count;
    result.expensive_collision_checks = expensive_collision_checks;
    result.dynamically_rejected_count = dynamically_rejected_count;
    result.dynamically_penalized_count = dynamically_penalized_count;
    result.unknown_horizon_count = unknown_horizon_count;

    auto t1 = std::chrono::high_resolution_clock::now();
    result.dwa_latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace dwa

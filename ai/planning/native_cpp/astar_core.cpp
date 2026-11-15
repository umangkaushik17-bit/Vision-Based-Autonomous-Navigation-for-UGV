#include "astar_core.h"
#include <queue>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <limits>

namespace astar {

struct AStarWorkspace {
    std::vector<uint32_t> visited_gen;
    std::vector<double> g_costs;
    std::vector<int32_t> came_from;
    std::vector<uint8_t> closed;
    uint32_t current_gen = 0;

    void prepare(size_t n) {
        if (visited_gen.size() < n) {
            visited_gen.assign(n, 0);
            g_costs.resize(n, std::numeric_limits<double>::infinity());
            came_from.resize(n, -1);
            closed.resize(n, 0);
            current_gen = 1;
        } else {
            current_gen++;
            if (current_gen == 0) {
                // Handle 32-bit integer overflow safely
                visited_gen.assign(n, 0);
                current_gen = 1;
            }
        }
    }
};

static thread_local AStarWorkspace tls_workspace;

struct AStarNode {
    double f;
    double g;
    int32_t r;
    int32_t c;
};

struct NodeComparator {
    bool operator()(const AStarNode& a, const AStarNode& b) const {
        if (a.f != b.f) return a.f > b.f;
        if (a.g != b.g) return a.g > b.g;
        if (a.r != b.r) return a.r > b.r;
        return a.c > b.c;
    }
};

struct Motion {
    int32_t dr;
    int32_t dc;
    double step_mult;
};

static const Motion MOTIONS[8] = {
    {-1, 0, 1.0},
    {1, 0, 1.0},
    {0, -1, 1.0},
    {0, 1, 1.0},
    {-1, -1, 1.4142135623730951},
    {-1, 1, 1.4142135623730951},
    {1, -1, 1.4142135623730951},
    {1, 1, 1.4142135623730951}
};

static inline double compute_heuristic(
    int32_t r, int32_t c,
    int32_t r_goal, int32_t c_goal,
    double res,
    int32_t type,
    double weight
) {
    double dr = std::abs(r_goal - r) * res;
    double dc = std::abs(c_goal - c) * res;
    double h = 0.0;
    if (type == 1) {
        // Euclidean distance
        h = std::hypot(dr, dc);
    } else {
        // Octile distance for 8-connected grid: never overestimates
        h = (dr + dc) + (1.4142135623730951 - 2.0) * std::min(dr, dc);
    }
    return h * weight;
}

void AStarPlannerCore::world_to_grid(
    const AStarGridMeta& meta,
    double x_m, double y_m,
    int32_t& row, int32_t& col
) const {
    if (meta.is_cartesian) {
        col = static_cast<int32_t>((x_m - meta.x_min) / meta.resolution_m);
        row = static_cast<int32_t>((y_m - meta.y_min) / meta.resolution_m);
    } else {
        // Standard top-down raster convention (WorldTerrainMap and BEVTerrainCostmap)
        row = static_cast<int32_t>((meta.x_max - x_m) / meta.resolution_m);
        col = static_cast<int32_t>((meta.y_max - y_m) / meta.resolution_m);
    }
}

void AStarPlannerCore::grid_to_world(
    const AStarGridMeta& meta,
    int32_t row, int32_t col,
    double& x_m, double& y_m
) const {
    if (meta.is_cartesian) {
        x_m = meta.x_min + (col + 0.5) * meta.resolution_m;
        y_m = meta.y_min + (row + 0.5) * meta.resolution_m;
    } else {
        x_m = meta.x_max - (row + 0.5) * meta.resolution_m;
        y_m = meta.y_max - (col + 0.5) * meta.resolution_m;
    }
}

bool AStarPlannerCore::is_in_bounds(
    const AStarGridMeta& meta,
    double x_m, double y_m
) const {
    return (x_m >= meta.x_min && x_m < meta.x_max &&
            y_m >= meta.y_min && y_m < meta.y_max);
}

double AStarPlannerCore::evaluate_cell_cost(
    int32_t r, int32_t c,
    const AStarCostConfig& cost_cfg,
    const AStarGridMeta& meta,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid,
    const uint8_t* unknown_grid,
    const float* traversability_grid,
    const float* slope_grid,
    const float* roughness_grid,
    const float* height_var_grid
) const {
    if (r < 0 || r >= meta.n_rows || c < 0 || c >= meta.n_cols) {
        return std::numeric_limits<double>::infinity();
    }
    int32_t idx = r * meta.n_cols + c;

    // 1. Hard Obstacle Precedence
    if (obstacle_grid && obstacle_grid[idx]) {
        return std::numeric_limits<double>::infinity();
    }

    // 2. Critical Hazard Precedence (hazard >= critical_hazard_threshold is lethal)
    if (hazard_grid && static_cast<double>(hazard_grid[idx]) >= cost_cfg.critical_hazard_threshold) {
        return std::numeric_limits<double>::infinity();
    }

    // 3. Lethal Cost Precedence (cost >= lethal_cost_threshold is lethal)
    double base_c = cost_grid ? static_cast<double>(cost_grid[idx]) : 0.0;
    if (base_c >= cost_cfg.lethal_cost_threshold) {
        return std::numeric_limits<double>::infinity();
    }

    // 4. Unknown Space Policy (Phase 12: UNKNOWN != SAFE)
    bool is_unknown = unknown_grid ? (unknown_grid[idx] != 0) : false;
    if (is_unknown && cost_cfg.unknown_is_lethal) {
        return std::numeric_limits<double>::infinity();
    }

    // Continuous weighted terrain cost formula
    double trav = traversability_grid ? static_cast<double>(traversability_grid[idx]) : 0.5;
    double haz = hazard_grid ? static_cast<double>(hazard_grid[idx]) : 0.0;
    double slope = slope_grid ? static_cast<double>(slope_grid[idx]) : 0.0;
    double rough = roughness_grid ? static_cast<double>(roughness_grid[idx]) : 0.0;
    double h_var = height_var_grid ? static_cast<double>(height_var_grid[idx]) : 0.0;
    double unk_val = is_unknown ? cost_cfg.unknown_cost_penalty : 0.0;

    // 5. Hard 2.5D Terrain Safety Cutoffs (Subphase 15.5: slope >= 28 deg or step >= 0.12 m)
    if (slope >= 28.0 || h_var >= cost_cfg.max_safe_step_m) {
        return std::numeric_limits<double>::infinity();
    }

    double cell_cost = (
        cost_cfg.w_cost * base_c +
        cost_cfg.w_traversability * (1.0 - trav) +
        cost_cfg.w_hazard * haz +
        cost_cfg.w_slope * std::min(slope / cost_cfg.max_safe_slope_deg, 2.0) +
        cost_cfg.w_roughness * rough +
        cost_cfg.w_height * std::min(h_var / cost_cfg.max_safe_step_m, 2.0) +
        cost_cfg.w_unknown * unk_val
    );

    return cell_cost;
}

bool AStarPlannerCore::is_point_collision(
    double x_m, double y_m,
    const AStarCostConfig& cost_cfg,
    const AStarGridMeta& meta,
    const AStarVehicleFootprint& vehicle,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid,
    const uint8_t* unknown_grid
) const {
    if (!is_in_bounds(meta, x_m, y_m)) return true;
    int32_t r, c;
    world_to_grid(meta, x_m, y_m, r, c);
    if (r < 0 || r >= meta.n_rows || c < 0 || c >= meta.n_cols) return true;
    int32_t idx = r * meta.n_cols + c;

    if (obstacle_grid && obstacle_grid[idx]) return true;
    if (hazard_grid && static_cast<double>(hazard_grid[idx]) >= cost_cfg.critical_hazard_threshold) return true;
    if (cost_cfg.unknown_is_lethal && unknown_grid && unknown_grid[idx]) return true;
    if (cost_grid && static_cast<double>(cost_grid[idx]) > vehicle.max_safe_cost) return true;

    return false;
}

bool AStarPlannerCore::is_circle_collision(
    double x_m, double y_m, double radius_m,
    const AStarCostConfig& cost_cfg,
    const AStarGridMeta& meta,
    const AStarVehicleFootprint& vehicle,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid,
    const uint8_t* unknown_grid
) const {
    if (!is_in_bounds(meta, x_m, y_m)) return true;
    double res = meta.resolution_m;
    int32_t r_cells = std::max(1, static_cast<int32_t>(std::ceil(radius_m / res)));
    int32_t r_center, c_center;
    world_to_grid(meta, x_m, y_m, r_center, c_center);

    int32_t r_min = std::max(0, r_center - r_cells);
    int32_t r_max = std::min(meta.n_rows - 1, r_center + r_cells);
    int32_t c_min = std::max(0, c_center - r_cells);
    int32_t c_max = std::min(meta.n_cols - 1, c_center + r_cells);

    double rad_sq = radius_m * radius_m;
    for (int32_t r = r_min; r <= r_max; ++r) {
        double dr = (r - r_center) * res;
        for (int32_t c = c_min; c <= c_max; ++c) {
            double dc = (c - c_center) * res;
            if (dr * dr + dc * dc <= rad_sq) {
                int32_t idx = r * meta.n_cols + c;
                if (obstacle_grid && obstacle_grid[idx]) return true;
                if (hazard_grid && static_cast<double>(hazard_grid[idx]) >= cost_cfg.critical_hazard_threshold) return true;
                if (cost_cfg.unknown_is_lethal && unknown_grid && unknown_grid[idx]) return true;
                if (cost_grid && static_cast<double>(cost_grid[idx]) > vehicle.max_safe_cost) return true;
            }
        }
    }
    return false;
}

bool AStarPlannerCore::is_footprint_collision(
    double x_m, double y_m, double yaw_rad,
    const AStarCostConfig& cost_cfg,
    const AStarGridMeta& meta,
    const AStarVehicleFootprint& vehicle,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid,
    const uint8_t* unknown_grid
) const {
    // 0. Ultra-fast O(1) center point check
    if (is_point_collision(x_m, y_m, cost_cfg, meta, vehicle,
                           obstacle_grid, cost_grid, hazard_grid, unknown_grid)) {
        return true;
    }

    double hl = (vehicle.length_m / 2.0) + vehicle.collision_margin_m;
    double hw = (vehicle.width_m / 2.0) + vehicle.collision_margin_m;
    double cos_y = std::cos(yaw_rad);
    double sin_y = std::sin(yaw_rad);

    double corners_x[4] = {
        x_m + (hl * cos_y - hw * sin_y),
        x_m + (hl * cos_y - (-hw) * sin_y),
        x_m + ((-hl) * cos_y - (-hw) * sin_y),
        x_m + ((-hl) * cos_y - hw * sin_y)
    };
    double corners_y[4] = {
        y_m + (hl * sin_y + hw * cos_y),
        y_m + (hl * sin_y + (-hw) * cos_y),
        y_m + ((-hl) * sin_y + (-hw) * cos_y),
        y_m + ((-hl) * sin_y + hw * cos_y)
    };

    // 1. Check 4 corners first (4 O(1) lookups)
    for (int32_t i = 0; i < 4; ++i) {
        if (is_point_collision(corners_x[i], corners_y[i], cost_cfg, meta, vehicle,
                               obstacle_grid, cost_grid, hazard_grid, unknown_grid)) {
            return true;
        }
    }

    // 2. Check oriented footprint perimeter samples
    double edge_step_m = 0.05;
    for (int32_t i = 0; i < 4; ++i) {
        int32_t next_i = (i + 1) % 4;
        double edge_dist = std::hypot(corners_x[next_i] - corners_x[i], corners_y[next_i] - corners_y[i]);
        int32_t n_sub = std::max(1, static_cast<int32_t>(std::ceil(edge_dist / edge_step_m)));
        for (int32_t k = 1; k < n_sub; ++k) {
            double alpha = static_cast<double>(k) / n_sub;
            double px = corners_x[i] + alpha * (corners_x[next_i] - corners_x[i]);
            double py = corners_y[i] + alpha * (corners_y[next_i] - corners_y[i]);
            if (is_point_collision(px, py, cost_cfg, meta, vehicle,
                                   obstacle_grid, cost_grid, hazard_grid, unknown_grid)) {
                return true;
            }
        }
    }

    // 3. Check interior sample grid (7 x 5)
    for (int32_t i = 0; i < 7; ++i) {
        double dx = -hl * 0.85 + (2.0 * hl * 0.85 * i) / 6.0;
        for (int32_t j = 0; j < 5; ++j) {
            double dy = -hw * 0.80 + (2.0 * hw * 0.80 * j) / 4.0;
            double ix = x_m + (dx * cos_y - dy * sin_y);
            double iy = y_m + (dx * sin_y + dy * cos_y);
            if (is_point_collision(ix, iy, cost_cfg, meta, vehicle,
                                   obstacle_grid, cost_grid, hazard_grid, unknown_grid)) {
                return true;
            }
        }
    }

    // 4. Inscribed Circle Check
    double r_inscribed = std::min(vehicle.length_m, vehicle.width_m) / 2.0;
    double eff_inscribed = r_inscribed + vehicle.collision_margin_m;
    if (is_circle_collision(x_m, y_m, eff_inscribed, cost_cfg, meta, vehicle,
                            obstacle_grid, cost_grid, hazard_grid, unknown_grid)) {
        return true;
    }

    return false;
}

bool AStarPlannerCore::is_segment_collision(
    double x1, double y1, double x2, double y2,
    double step_m, bool check_footprint, double yaw_rad,
    const AStarCostConfig& cost_cfg,
    const AStarGridMeta& meta,
    const AStarVehicleFootprint& vehicle,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid,
    const uint8_t* unknown_grid
) const {
    double dx = x2 - x1;
    double dy = y2 - y1;
    double dist = std::hypot(dx, dy);

    if (dist < 1e-4) {
        if (check_footprint) {
            return is_footprint_collision(x1, y1, yaw_rad, cost_cfg, meta, vehicle,
                                          obstacle_grid, cost_grid, hazard_grid, unknown_grid);
        }
        return is_point_collision(x1, y1, cost_cfg, meta, vehicle,
                                  obstacle_grid, cost_grid, hazard_grid, unknown_grid);
    }

    double seg_yaw = (std::abs(yaw_rad) > 1e-4) ? yaw_rad : std::atan2(dy, dx);
    int32_t steps = std::max(2, static_cast<int32_t>(std::ceil(dist / step_m)));

    // Fast pass 1: Check centerline points first (midpoint, then full centerline) in O(steps)
    int32_t mid = steps / 2;
    if (is_point_collision(x1 + (static_cast<double>(mid) / steps) * dx,
                           y1 + (static_cast<double>(mid) / steps) * dy,
                           cost_cfg, meta, vehicle,
                           obstacle_grid, cost_grid, hazard_grid, unknown_grid)) {
        return true;
    }
    for (int32_t s = 0; s <= steps; ++s) {
        double alpha = static_cast<double>(s) / steps;
        if (is_point_collision(x1 + alpha * dx, y1 + alpha * dy,
                               cost_cfg, meta, vehicle,
                               obstacle_grid, cost_grid, hazard_grid, unknown_grid)) {
            return true;
        }
    }

    if (!check_footprint) {
        return false;
    }

    // Fast pass 2: Check midpoint footprint first, then remaining samples
    if (is_footprint_collision(x1 + (static_cast<double>(mid) / steps) * dx,
                               y1 + (static_cast<double>(mid) / steps) * dy,
                               seg_yaw, cost_cfg, meta, vehicle,
                               obstacle_grid, cost_grid, hazard_grid, unknown_grid)) {
        return true;
    }

    for (int32_t s = 0; s <= steps; ++s) {
        if (s == mid) continue;
        double alpha = static_cast<double>(s) / steps;
        double x = x1 + alpha * dx;
        double y = y1 + alpha * dy;
        if (is_footprint_collision(x, y, seg_yaw, cost_cfg, meta, vehicle,
                                   obstacle_grid, cost_grid, hazard_grid, unknown_grid)) {
            return true;
        }
    }
    return false;
}

bool AStarPlannerCore::find_nearest_free_point(
    double gx, double gy, double goal_yaw,
    double search_radius_m, bool check_footprint,
    const AStarCostConfig& cost_cfg,
    const AStarGridMeta& meta,
    const AStarVehicleFootprint& vehicle,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid,
    const uint8_t* unknown_grid,
    const uint8_t* infl_obs,
    const uint8_t* infl_haz,
    double& out_gx, double& out_gy
) const {
    double res = meta.resolution_m;
    int32_t max_steps = std::max(1, static_cast<int32_t>(std::ceil(search_radius_m / res)));
    double best_d = std::numeric_limits<double>::infinity();
    bool found = false;

    for (int32_t sx = -max_steps; sx <= max_steps; ++sx) {
        for (int32_t sy = -max_steps; sy <= max_steps; ++sy) {
            double d = std::hypot(sx * res, sy * res);
            if (d > search_radius_m || d >= best_d) continue;

            double cand_x = gx + sx * res;
            double cand_y = gy + sy * res;
            if (!is_in_bounds(meta, cand_x, cand_y)) continue;

            int32_t cr, cc;
            world_to_grid(meta, cand_x, cand_y, cr, cc);
            if (cr < 0 || cr >= meta.n_rows || cc < 0 || cc >= meta.n_cols) continue;

            int32_t idx = cr * meta.n_cols + cc;
            if (infl_obs && infl_obs[idx]) continue;
            if (infl_haz && infl_haz[idx]) continue;
            if (is_point_collision(cand_x, cand_y, cost_cfg, meta, vehicle,
                                   obstacle_grid, cost_grid, hazard_grid, unknown_grid)) continue;
            if (check_footprint && is_footprint_collision(cand_x, cand_y, goal_yaw, cost_cfg, meta, vehicle,
                                                          obstacle_grid, cost_grid, hazard_grid, unknown_grid)) continue;

            best_d = d;
            out_gx = cand_x;
            out_gy = cand_y;
            found = true;
        }
    }
    return found;
}

std::vector<std::pair<double, double>> AStarPlannerCore::prune_collinear(
    const std::vector<std::pair<double, double>>& path_xy
) const {
    if (path_xy.size() <= 2) return path_xy;

    std::vector<std::pair<double, double>> pruned;
    pruned.push_back(path_xy[0]);

    for (size_t i = 1; i < path_xy.size() - 1; ++i) {
        const auto& p_prev = pruned.back();
        const auto& p_curr = path_xy[i];
        const auto& p_next = path_xy[i + 1];

        double d1_x = p_curr.first - p_prev.first;
        double d1_y = p_curr.second - p_prev.second;
        double d2_x = p_next.first - p_curr.first;
        double d2_y = p_next.second - p_curr.second;

        double cross = std::abs(d1_x * d2_y - d1_y * d2_x);
        if (cross > 1e-4) {
            pruned.push_back(p_curr);
        }
    }

    pruned.push_back(path_xy.back());
    return pruned;
}

std::vector<std::pair<double, double>> AStarPlannerCore::smooth_path(
    const std::vector<std::pair<double, double>>& raw_xy,
    const AStarCostConfig& cost_cfg,
    const AStarGridMeta& meta,
    const AStarVehicleFootprint& vehicle,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid,
    const uint8_t* unknown_grid
) const {
    if (raw_xy.size() <= 2) return raw_xy;

    // 1. Collinear pruning first
    auto pruned = prune_collinear(raw_xy);
    if (pruned.size() <= 2) return pruned;

    // 2. Line-of-sight shortcutting with footprint validation
    std::vector<std::pair<double, double>> smoothed;
    smoothed.push_back(pruned[0]);
    int32_t curr_idx = 0;
    int32_t n = static_cast<int32_t>(pruned.size());
    int32_t iters = 0;

    double step_m = (cost_cfg.smoothing_step_m > 0.0) ? cost_cfg.smoothing_step_m : 0.08;

    while (curr_idx < n - 1 && iters < cost_cfg.max_smoothing_iterations) {
        iters++;
        int32_t next_idx = n - 1;
        while (next_idx > curr_idx + 1) {
            double p1_x = pruned[curr_idx].first;
            double p1_y = pruned[curr_idx].second;
            double p2_x = pruned[next_idx].first;
            double p2_y = pruned[next_idx].second;

            AStarVehicleFootprint eff_vehicle = vehicle;
            if (cost_grid) {
                int32_t r1, c1, r2, c2;
                world_to_grid(meta, p1_x, p1_y, r1, c1);
                world_to_grid(meta, p2_x, p2_y, r2, c2);
                r1 = std::clamp(r1, 0, meta.n_rows - 1);
                c1 = std::clamp(c1, 0, meta.n_cols - 1);
                r2 = std::clamp(r2, 0, meta.n_rows - 1);
                c2 = std::clamp(c2, 0, meta.n_cols - 1);
                double end_cost = std::max(static_cast<double>(cost_grid[r1 * meta.n_cols + c1]),
                                           static_cast<double>(cost_grid[r2 * meta.n_cols + c2]));
                eff_vehicle.max_safe_cost = std::min(vehicle.max_safe_cost, end_cost + 0.12);
            }

            if (!is_segment_collision(p1_x, p1_y, p2_x, p2_y, step_m, true, 0.0,
                                      cost_cfg, meta, eff_vehicle,
                                      obstacle_grid, cost_grid, hazard_grid, unknown_grid)) {
                break;
            }
            next_idx--;
        }
        smoothed.push_back(pruned[next_idx]);
        curr_idx = next_idx;
    }

    return smoothed;
}

int AStarPlannerCore::plan(
    double start_x, double start_y,
    double goal_x, double goal_y, double goal_yaw,
    const AStarCostConfig& cost_cfg,
    const AStarGridMeta& grid_meta,
    const AStarVehicleFootprint& vehicle,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid,
    const uint8_t* unknown_grid,
    const float* traversability_grid,
    const float* slope_grid,
    const float* roughness_grid,
    const float* height_var_grid,
    const uint8_t* infl_obstacle_grid,
    const uint8_t* infl_hazard_grid,
    AStarResult& out_result,
    double* out_raw_x,
    double* out_raw_y,
    int32_t max_raw,
    double* out_smooth_x,
    double* out_smooth_y,
    int32_t max_smooth
) {
    auto t0 = std::chrono::high_resolution_clock::now();

    out_result.status = 1; // Default NO_PATH
    out_result.raw_waypoint_count = 0;
    out_result.smoothed_waypoint_count = 0;
    out_result.nodes_explored = 0;
    out_result.path_cost = 0.0;
    out_result.path_length_m = 0.0;
    out_result.planning_latency_ms = 0.0;
    out_result.smoothing_latency_ms = 0.0;

    // 1. Bounds Validation
    if (!is_in_bounds(grid_meta, start_x, start_y)) {
        out_result.status = 2; // INVALID_START
        return 2;
    }
    if (!is_in_bounds(grid_meta, goal_x, goal_y)) {
        out_result.status = 3; // INVALID_GOAL
        return 3;
    }

    int32_t r_start, c_start;
    world_to_grid(grid_meta, start_x, start_y, r_start, c_start);

    int32_t r_goal, c_goal;
    world_to_grid(grid_meta, goal_x, goal_y, r_goal, c_goal);

    // 2. Start / Goal Collision Validation
    if (is_point_collision(start_x, start_y, cost_cfg, grid_meta, vehicle,
                           obstacle_grid, cost_grid, hazard_grid, unknown_grid)) {
        out_result.status = 4; // START_BLOCKED
        return 4;
    }

    int32_t goal_idx = r_goal * grid_meta.n_cols + c_goal;
    bool goal_in_infl = (infl_obstacle_grid && infl_obstacle_grid[goal_idx]) ||
                        (infl_hazard_grid && infl_hazard_grid[goal_idx]);

    if (is_point_collision(goal_x, goal_y, cost_cfg, grid_meta, vehicle,
                           obstacle_grid, cost_grid, hazard_grid, unknown_grid) ||
        is_footprint_collision(goal_x, goal_y, goal_yaw, cost_cfg, grid_meta, vehicle,
                               obstacle_grid, cost_grid, hazard_grid, unknown_grid) ||
        goal_in_infl) {

        double free_gx = goal_x, free_gy = goal_y;
        if (find_nearest_free_point(goal_x, goal_y, goal_yaw, 1.5, true,
                                    cost_cfg, grid_meta, vehicle,
                                    obstacle_grid, cost_grid, hazard_grid, unknown_grid,
                                    infl_obstacle_grid, infl_hazard_grid,
                                    free_gx, free_gy)) {
            goal_x = free_gx;
            goal_y = free_gy;
            world_to_grid(grid_meta, goal_x, goal_y, r_goal, c_goal);
        } else {
            out_result.status = 5; // GOAL_BLOCKED
            return 5;
        }
    }

    // Direct endpoint check: Start and Goal in same cell
    if (r_start == r_goal && c_start == c_goal) {
        if (max_raw >= 2 && out_raw_x && out_raw_y) {
            out_raw_x[0] = start_x; out_raw_y[0] = start_y;
            out_raw_x[1] = goal_x;  out_raw_y[1] = goal_y;
        }
        if (max_smooth >= 2 && out_smooth_x && out_smooth_y) {
            out_smooth_x[0] = start_x; out_smooth_y[0] = start_y;
            out_smooth_x[1] = goal_x;  out_smooth_y[1] = goal_y;
        }
        out_result.status = 0; // SUCCESS
        out_result.raw_waypoint_count = 2;
        out_result.smoothed_waypoint_count = 2;
        out_result.nodes_explored = 1;
        out_result.path_cost = 0.0;
        out_result.path_length_m = std::hypot(goal_x - start_x, goal_y - start_y);
        auto t_end = std::chrono::high_resolution_clock::now();
        out_result.planning_latency_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
        return 0;
    }

    // 3. Corridor Bounding Box
    double res = grid_meta.resolution_m;
    double bbox_margin_m = 12.0;
    int32_t bbox_margin_cells = std::max(30, static_cast<int32_t>(std::ceil(bbox_margin_m / res)));

    int32_t r_min = std::max(0, std::min(r_start, r_goal) - bbox_margin_cells);
    int32_t r_max = std::min(grid_meta.n_rows - 1, std::max(r_start, r_goal) + bbox_margin_cells);
    int32_t c_min = std::max(0, std::min(c_start, c_goal) - bbox_margin_cells);
    int32_t c_max = std::min(grid_meta.n_cols - 1, std::max(c_start, c_goal) + bbox_margin_cells);

    // 4. Thread-Local Workspace Preparation (Zero Allocation per Replan)
    size_t total_cells = static_cast<size_t>(grid_meta.n_rows) * grid_meta.n_cols;
    tls_workspace.prepare(total_cells);
    uint32_t gen = tls_workspace.current_gen;

    std::priority_queue<AStarNode, std::vector<AStarNode>, NodeComparator> open_set;

    double start_h = compute_heuristic(r_start, c_start, r_goal, c_goal, res,
                                       cost_cfg.heuristic_type, cost_cfg.heuristic_weight);

    int32_t start_idx = r_start * grid_meta.n_cols + c_start;
    tls_workspace.visited_gen[start_idx] = gen;
    tls_workspace.g_costs[start_idx] = 0.0;
    tls_workspace.came_from[start_idx] = -1;
    tls_workspace.closed[start_idx] = 0;

    open_set.push({start_h, 0.0, r_start, c_start});

    int32_t nodes_explored = 0;
    bool goal_reached = false;
    int32_t goal_reached_idx = -1;

    // 5. Main Graph Search Loop
    while (!open_set.empty()) {
        AStarNode curr = open_set.top();
        open_set.pop();

        int32_t curr_idx = curr.r * grid_meta.n_cols + curr.c;

        if (tls_workspace.closed[curr_idx]) {
            continue;
        }
        tls_workspace.closed[curr_idx] = 1;
        nodes_explored++;

        if (curr.r == r_goal && curr.c == c_goal) {
            goal_reached = true;
            goal_reached_idx = curr_idx;
            break;
        }

        // Optional JPS parent direction extraction for natural/forced neighbor pruning
        int32_t parent_idx = tls_workspace.came_from[curr_idx];
        int32_t p_dr = 0;
        int32_t p_dc = 0;
        if (cost_cfg.enable_jps && parent_idx >= 0) {
            int32_t pr = parent_idx / grid_meta.n_cols;
            int32_t pc = parent_idx % grid_meta.n_cols;
            p_dr = (curr.r > pr) ? 1 : ((curr.r < pr) ? -1 : 0);
            p_dc = (curr.c > pc) ? 1 : ((curr.c < pc) ? -1 : 0);
        }

        // Explore 8 neighbors
        for (int32_t m = 0; m < 8; ++m) {
            int32_t dr = MOTIONS[m].dr;
            int32_t dc = MOTIONS[m].dc;

            // JPS directional symmetry pruning (skip dominated backward/orthogonal neighbors unless forced)
            if (cost_cfg.enable_jps && parent_idx >= 0) {
                int32_t dot = dr * p_dr + dc * p_dc;
                if (dot < 0) {
                    continue;
                }
            }

            int32_t nr = curr.r + dr;
            int32_t nc = curr.c + dc;

            if (nr < r_min || nr > r_max || nc < c_min || nc > c_max) {
                continue;
            }

            int32_t n_idx = nr * grid_meta.n_cols + nc;
            if (tls_workspace.visited_gen[n_idx] == gen && tls_workspace.closed[n_idx]) {
                continue;
            }

            // Chassis clearance check against dilated obstacle/hazard masks
            bool is_goal_cell = (nr == r_goal && nc == c_goal);
            if (!is_goal_cell) {
                if (infl_obstacle_grid && infl_obstacle_grid[n_idx]) {
                    continue;
                }
                if (infl_hazard_grid && infl_hazard_grid[n_idx]) {
                    continue;
                }
            }

            // Cell terrain cost and hard safety check
            double cell_terrain = evaluate_cell_cost(
                nr, nc, cost_cfg, grid_meta,
                obstacle_grid, cost_grid, hazard_grid, unknown_grid,
                traversability_grid, slope_grid, roughness_grid, height_var_grid
            );

            if (std::isinf(cell_terrain)) {
                continue;
            }

            double step_cost = res * MOTIONS[m].step_mult * (1.0 + cell_terrain);
            double tentative_g = curr.g + step_cost;

            double existing_g = (tls_workspace.visited_gen[n_idx] == gen)
                                ? tls_workspace.g_costs[n_idx]
                                : std::numeric_limits<double>::infinity();

            if (tentative_g < existing_g) {
                tls_workspace.visited_gen[n_idx] = gen;
                tls_workspace.g_costs[n_idx] = tentative_g;
                tls_workspace.came_from[n_idx] = curr_idx;
                tls_workspace.closed[n_idx] = 0;

                double h = compute_heuristic(nr, nc, r_goal, c_goal, res,
                                             cost_cfg.heuristic_type, cost_cfg.heuristic_weight);
                open_set.push({tentative_g + h, tentative_g, nr, nc});
            }
        }
    }

    auto t_search_end = std::chrono::high_resolution_clock::now();
    out_result.nodes_explored = nodes_explored;
    out_result.planning_latency_ms = std::chrono::duration<double, std::milli>(t_search_end - t0).count();

    if (!goal_reached) {
        out_result.status = 1; // NO_PATH
        return 1;
    }

    // 6. Path Reconstruction
    std::vector<std::pair<double, double>> raw_coords;
    int32_t curr = goal_reached_idx;
    while (curr != -1) {
        int32_t r = curr / grid_meta.n_cols;
        int32_t c = curr % grid_meta.n_cols;
        double xm, ym;
        grid_to_world(grid_meta, r, c, xm, ym);
        raw_coords.push_back({xm, ym});
        curr = tls_workspace.came_from[curr];
    }
    std::reverse(raw_coords.begin(), raw_coords.end());

    // Ensure exact endpoints
    raw_coords.front() = {start_x, start_y};
    raw_coords.back() = {goal_x, goal_y};

    out_result.path_cost = tls_workspace.g_costs[goal_reached_idx];
    out_result.raw_waypoint_count = static_cast<int32_t>(raw_coords.size());

    // Calculate raw path arc length
    double total_len = 0.0;
    for (size_t i = 1; i < raw_coords.size(); ++i) {
        total_len += std::hypot(raw_coords[i].first - raw_coords[i - 1].first,
                                raw_coords[i].second - raw_coords[i - 1].second);
    }
    out_result.path_length_m = total_len;

    // Copy raw waypoints to output buffer
    int32_t n_raw_copy = std::min(max_raw, out_result.raw_waypoint_count);
    if (out_raw_x && out_raw_y) {
        for (int32_t i = 0; i < n_raw_copy; ++i) {
            out_raw_x[i] = raw_coords[i].first;
            out_raw_y[i] = raw_coords[i].second;
        }
    }

    // 7. Path Smoothing & Footprint Clearance
    if (cost_cfg.enable_smoothing) {
        auto t_smooth_start = std::chrono::high_resolution_clock::now();
        auto smoothed = smooth_path(raw_coords, cost_cfg, grid_meta, vehicle,
                                    obstacle_grid, cost_grid, hazard_grid, unknown_grid);
        auto t_smooth_end = std::chrono::high_resolution_clock::now();
        out_result.smoothing_latency_ms = std::chrono::duration<double, std::milli>(t_smooth_end - t_smooth_start).count();

        out_result.smoothed_waypoint_count = static_cast<int32_t>(smoothed.size());

        // Update smoothed length
        double s_len = 0.0;
        for (size_t i = 1; i < smoothed.size(); ++i) {
            s_len += std::hypot(smoothed[i].first - smoothed[i - 1].first,
                                smoothed[i].second - smoothed[i - 1].second);
        }
        out_result.path_length_m = s_len;

        int32_t n_smooth_copy = std::min(max_smooth, out_result.smoothed_waypoint_count);
        if (out_smooth_x && out_smooth_y) {
            for (int32_t i = 0; i < n_smooth_copy; ++i) {
                out_smooth_x[i] = smoothed[i].first;
                out_smooth_y[i] = smoothed[i].second;
            }
        }
    } else {
        out_result.smoothed_waypoint_count = out_result.raw_waypoint_count;
        int32_t n_smooth_copy = std::min(max_smooth, out_result.raw_waypoint_count);
        if (out_smooth_x && out_smooth_y) {
            for (int32_t i = 0; i < n_smooth_copy; ++i) {
                out_smooth_x[i] = raw_coords[i].first;
                out_smooth_y[i] = raw_coords[i].second;
            }
        }
    }

    out_result.status = 0; // SUCCESS
    return 0;
}

} // namespace astar

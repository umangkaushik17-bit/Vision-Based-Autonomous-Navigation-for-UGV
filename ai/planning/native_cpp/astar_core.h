#pragma once

#include "astar_types.h"
#include <vector>
#include <utility>

namespace astar {

class AStarPlannerCore {
public:
    AStarPlannerCore() = default;
    ~AStarPlannerCore() = default;

    int plan(
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
    );

    double evaluate_cell_cost(
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
    ) const;

    bool is_footprint_collision(
        double x_m, double y_m, double yaw_rad,
        const AStarCostConfig& cost_cfg,
        const AStarGridMeta& meta,
        const AStarVehicleFootprint& vehicle,
        const uint8_t* obstacle_grid,
        const float* cost_grid,
        const float* hazard_grid,
        const uint8_t* unknown_grid
    ) const;

    bool is_point_collision(
        double x_m, double y_m,
        const AStarCostConfig& cost_cfg,
        const AStarGridMeta& meta,
        const AStarVehicleFootprint& vehicle,
        const uint8_t* obstacle_grid,
        const float* cost_grid,
        const float* hazard_grid,
        const uint8_t* unknown_grid
    ) const;

    bool is_circle_collision(
        double x_m, double y_m, double radius_m,
        const AStarCostConfig& cost_cfg,
        const AStarGridMeta& meta,
        const AStarVehicleFootprint& vehicle,
        const uint8_t* obstacle_grid,
        const float* cost_grid,
        const float* hazard_grid,
        const uint8_t* unknown_grid
    ) const;

    bool is_segment_collision(
        double x1, double y1, double x2, double y2,
        double step_m, bool check_footprint, double yaw_rad,
        const AStarCostConfig& cost_cfg,
        const AStarGridMeta& meta,
        const AStarVehicleFootprint& vehicle,
        const uint8_t* obstacle_grid,
        const float* cost_grid,
        const float* hazard_grid,
        const uint8_t* unknown_grid
    ) const;

private:
    void world_to_grid(
        const AStarGridMeta& meta,
        double x_m, double y_m,
        int32_t& row, int32_t& col
    ) const;

    void grid_to_world(
        const AStarGridMeta& meta,
        int32_t row, int32_t col,
        double& x_m, double& y_m
    ) const;

    bool is_in_bounds(
        const AStarGridMeta& meta,
        double x_m, double y_m
    ) const;

    bool find_nearest_free_point(
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
    ) const;

    std::vector<std::pair<double, double>> smooth_path(
        const std::vector<std::pair<double, double>>& raw_xy,
        const AStarCostConfig& cost_cfg,
        const AStarGridMeta& meta,
        const AStarVehicleFootprint& vehicle,
        const uint8_t* obstacle_grid,
        const float* cost_grid,
        const float* hazard_grid,
        const uint8_t* unknown_grid
    ) const;

    std::vector<std::pair<double, double>> prune_collinear(
        const std::vector<std::pair<double, double>>& path_xy
    ) const;
};

} // namespace astar

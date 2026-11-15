#pragma once

#include "astar_types.h"

#ifdef _WIN32
#define ASTAR_EXPORT __declspec(dllexport)
#else
#define ASTAR_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

ASTAR_EXPORT int astar_plan_c(
    double start_x, double start_y,
    double goal_x, double goal_y, double goal_yaw,
    const astar::AStarCostConfig* cost_cfg,
    const astar::AStarGridMeta* grid_meta,
    const astar::AStarVehicleFootprint* vehicle,
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
    astar::AStarResult* out_result,
    double* out_raw_x,
    double* out_raw_y,
    int32_t max_raw,
    double* out_smooth_x,
    double* out_smooth_y,
    int32_t max_smooth
);

ASTAR_EXPORT double astar_evaluate_cell_cost_c(
    int32_t r, int32_t c,
    const astar::AStarCostConfig* cost_cfg,
    const astar::AStarGridMeta* meta,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid,
    const uint8_t* unknown_grid,
    const float* traversability_grid,
    const float* slope_grid,
    const float* roughness_grid,
    const float* height_var_grid
);

ASTAR_EXPORT int astar_is_footprint_collision_c(
    double x_m, double y_m, double yaw_rad,
    const astar::AStarCostConfig* cost_cfg,
    const astar::AStarGridMeta* meta,
    const astar::AStarVehicleFootprint* vehicle,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid,
    const uint8_t* unknown_grid
);

ASTAR_EXPORT int astar_is_segment_collision_c(
    double x1, double y1, double x2, double y2,
    double step_m, int32_t check_footprint, double yaw_rad,
    const astar::AStarCostConfig* cost_cfg,
    const astar::AStarGridMeta* meta,
    const astar::AStarVehicleFootprint* vehicle,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid,
    const uint8_t* unknown_grid
);

}

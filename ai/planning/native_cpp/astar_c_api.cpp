#include "astar_c_api.h"
#include "astar_core.h"
#include <cstring>

static thread_local astar::AStarPlannerCore tl_astar_planner;

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
) {
    if (!cost_cfg || !grid_meta || !vehicle || !out_result) {
        return -1;
    }

    return tl_astar_planner.plan(
        start_x, start_y,
        goal_x, goal_y, goal_yaw,
        *cost_cfg,
        *grid_meta,
        *vehicle,
        obstacle_grid,
        cost_grid,
        hazard_grid,
        unknown_grid,
        traversability_grid,
        slope_grid,
        roughness_grid,
        height_var_grid,
        infl_obstacle_grid,
        infl_hazard_grid,
        *out_result,
        out_raw_x,
        out_raw_y,
        max_raw,
        out_smooth_x,
        out_smooth_y,
        max_smooth
    );
}

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
) {
    if (!cost_cfg || !meta) {
        return 1e12; // infinity
    }
    return tl_astar_planner.evaluate_cell_cost(
        r, c,
        *cost_cfg,
        *meta,
        obstacle_grid,
        cost_grid,
        hazard_grid,
        unknown_grid,
        traversability_grid,
        slope_grid,
        roughness_grid,
        height_var_grid
    );
}

ASTAR_EXPORT int astar_is_footprint_collision_c(
    double x_m, double y_m, double yaw_rad,
    const astar::AStarCostConfig* cost_cfg,
    const astar::AStarGridMeta* meta,
    const astar::AStarVehicleFootprint* vehicle,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid,
    const uint8_t* unknown_grid
) {
    if (!cost_cfg || !meta || !vehicle) {
        return 1;
    }
    return tl_astar_planner.is_footprint_collision(
        x_m, y_m, yaw_rad,
        *cost_cfg,
        *meta,
        *vehicle,
        obstacle_grid,
        cost_grid,
        hazard_grid,
        unknown_grid
    ) ? 1 : 0;
}

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
) {
    if (!cost_cfg || !meta || !vehicle) {
        return 1;
    }
    return tl_astar_planner.is_segment_collision(
        x1, y1, x2, y2,
        step_m, (check_footprint != 0), yaw_rad,
        *cost_cfg,
        *meta,
        *vehicle,
        obstacle_grid,
        cost_grid,
        hazard_grid,
        unknown_grid
    ) ? 1 : 0;
}

}

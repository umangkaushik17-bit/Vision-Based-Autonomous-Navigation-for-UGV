#include "dwa_c_api.h"
#include "dwa_core.h"
#include <array>

extern "C" {

DWA_EXPORT int dwa_plan_c(
    const dwa::RobotState* state,
    const dwa::VehicleConfig* vehicle,
    const dwa::DWAConfig* dwa_cfg,
    const dwa::ScoringWeights* weights,
    const double* path_x,
    const double* path_y,
    int32_t path_len,
    double local_target_x,
    double local_target_y,
    double d_curr,
    const dwa::CostmapMeta* costmap_meta,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid,
    const uint8_t* unknown_grid,
    const float* traversability_grid,
    const float* slope_grid,
    const float* roughness_grid,
    const float* height_var_grid,
    const dwa::DynamicConfig* dyn_cfg,
    const dwa::ObstacleTrack* tracks,
    int32_t num_tracks,
    int32_t is_degraded,
    dwa::DWAResult* out_result,
    dwa::CandidateDiagnostic* out_diagnostics,
    double* out_trajectories
) {
    if (!state || !vehicle || !dwa_cfg || !weights || !costmap_meta || !out_result || !out_diagnostics) {
        return -1;
    }

    try {
        dwa::DWAPlannerCore planner(*vehicle, *dwa_cfg, *weights);
        dwa::DynamicConfig def_dyn{};
        if (dyn_cfg) {
            def_dyn = *dyn_cfg;
        }

        planner.plan(
            *state,
            path_x,
            path_y,
            path_len,
            local_target_x,
            local_target_y,
            d_curr,
            *costmap_meta,
            obstacle_grid,
            cost_grid,
            hazard_grid,
            unknown_grid,
            traversability_grid,
            slope_grid,
            roughness_grid,
            height_var_grid,
            def_dyn,
            tracks,
            num_tracks,
            is_degraded != 0,
            *out_result,
            out_diagnostics,
            out_trajectories
        );
        return 0;
    } catch (...) {
        return -2;
    }
}

DWA_EXPORT int dwa_sat_intersect_c(
    const double* poly1_x,
    const double* poly1_y,
    const double* poly2_x,
    const double* poly2_y
) {
    if (!poly1_x || !poly1_y || !poly2_x || !poly2_y) {
        return -1;
    }

    std::array<std::pair<double, double>, 4> p1 = {{
        {poly1_x[0], poly1_y[0]},
        {poly1_x[1], poly1_y[1]},
        {poly1_x[2], poly1_y[2]},
        {poly1_x[3], poly1_y[3]}
    }};

    std::array<std::pair<double, double>, 4> p2 = {{
        {poly2_x[0], poly2_y[0]},
        {poly2_x[1], poly2_y[1]},
        {poly2_x[2], poly2_y[2]},
        {poly2_x[3], poly2_y[3]}
    }};

    return dwa::DWAPlannerCore::sat_polygon_intersect(p1, p2) ? 1 : 0;
}

DWA_EXPORT double dwa_polygon_distance_c(
    const double* poly1_x,
    const double* poly1_y,
    const double* poly2_x,
    const double* poly2_y
) {
    if (!poly1_x || !poly1_y || !poly2_x || !poly2_y) {
        return -1.0;
    }

    std::array<std::pair<double, double>, 4> p1 = {{
        {poly1_x[0], poly1_y[0]},
        {poly1_x[1], poly1_y[1]},
        {poly1_x[2], poly1_y[2]},
        {poly1_x[3], poly1_y[3]}
    }};

    std::array<std::pair<double, double>, 4> p2 = {{
        {poly2_x[0], poly2_y[0]},
        {poly2_x[1], poly2_y[1]},
        {poly2_x[2], poly2_y[2]},
        {poly2_x[3], poly2_y[3]}
    }};

    return dwa::DWAPlannerCore::polygon_polygon_distance(p1, p2);
}

DWA_EXPORT int dwa_footprint_collision_c(
    double x, double y, double yaw,
    const dwa::VehicleConfig* vehicle,
    const dwa::CostmapMeta* costmap_meta,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid
) {
    if (!vehicle || !costmap_meta) {
        return -1;
    }
    try {
        dwa::DWAConfig dwa_cfg{};
        dwa::ScoringWeights weights{};
        dwa::DWAPlannerCore planner(*vehicle, dwa_cfg, weights);
        bool col = planner.is_footprint_collision(x, y, yaw, *costmap_meta, obstacle_grid, cost_grid, hazard_grid);
        return col ? 1 : 0;
    } catch (...) {
        return -2;
    }
}

}

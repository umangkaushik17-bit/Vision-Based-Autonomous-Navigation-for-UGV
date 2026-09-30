#pragma once

#include "dwa_types.h"

#ifdef _WIN32
#define DWA_EXPORT __declspec(dllexport)
#else
#define DWA_EXPORT __attribute__((visibility("default")))
#endif

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
);

DWA_EXPORT int dwa_sat_intersect_c(
    const double* poly1_x,
    const double* poly1_y,
    const double* poly2_x,
    const double* poly2_y
);

DWA_EXPORT double dwa_polygon_distance_c(
    const double* poly1_x,
    const double* poly1_y,
    const double* poly2_x,
    const double* poly2_y
);

DWA_EXPORT int dwa_footprint_collision_c(
    double x, double y, double yaw,
    const dwa::VehicleConfig* vehicle,
    const dwa::CostmapMeta* costmap_meta,
    const uint8_t* obstacle_grid,
    const float* cost_grid,
    const float* hazard_grid
);

}

#pragma once

#include "dwa_types.h"
#include <vector>
#include <array>
#include <utility>

namespace dwa {

class DWAPlannerCore {
public:
    DWAPlannerCore(
        const VehicleConfig& vehicle,
        const DWAConfig& dwa_cfg,
        const ScoringWeights& weights
    );

    void plan(
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
        // Outputs
        DWAResult& result,
        CandidateDiagnostic* candidate_diagnostics, // Array of size v_samples * omega_samples
        double* out_trajectories // Array of size (K * S * 3) [x, y, yaw]
    );

    // Dynamic window calculation
    void compute_dynamic_window(
        double curr_v,
        double curr_omega,
        double dt,
        bool is_degraded,
        double& v_low,
        double& v_high,
        double& w_low,
        double& w_high
    ) const;

    // Fast 8-axis Separating Axis Theorem (SAT) convex polygon intersection
    static bool sat_polygon_intersect(
        const std::array<std::pair<double, double>, 4>& poly1,
        const std::array<std::pair<double, double>, 4>& poly2
    );

    // Minimum distance between two convex 4-vertex polygons
    static double polygon_polygon_distance(
        const std::array<std::pair<double, double>, 4>& poly1,
        const std::array<std::pair<double, double>, 4>& poly2
    );

    // Fast static footprint collision check at a single pose (x, y, yaw)
    bool is_footprint_collision(
        double x, double y, double yaw,
        const CostmapMeta& costmap_meta,
        const uint8_t* obstacle_grid,
        const float* cost_grid,
        const float* hazard_grid
    ) const;

private:
    VehicleConfig vehicle_;
    DWAConfig dwa_cfg_;
    ScoringWeights weights_;

    int32_t steps_;
    std::vector<std::pair<double, double>> footprint_offsets_; // Precomputed body offsets

    void precompute_footprint_offsets();

    static double point_to_segment_dist(
        double px, double py,
        double x1, double y1,
        double x2, double y2
    );

    bool interpolate_track_pose(
        const ObstacleTrack& track,
        double query_t,
        double& out_x,
        double& out_y,
        double& out_yaw,
        double& out_unc
    ) const;
};

} // namespace dwa

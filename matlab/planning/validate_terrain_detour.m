function res = validate_terrain_detour()
% VALIDATE_TERRAIN_DETOUR Reproduces and analyzes the Section 61 terrain detour scenario:
% Direct Route (crosses 30 deg lethal slope) vs. A* Selected Detour (routes through safe terrain).

    fprintf('--- Validating Section 61 Terrain Detour Scenario ---\n');

    d_traj = load('matlab/data/planner_trajectories.mat');
    w_astar = double(d_traj.astar_detour.waypoints);
    w_direct = double(d_traj.direct_straight_line.waypoints);

    len_astar = double(d_traj.astar_detour.length_m);
    len_direct = double(d_traj.direct_straight_line.length_m);

    % Load detour costmap to sample terrain slope and cost along both paths
    d_cm = load('matlab/data/costmap_scenarios.mat');
    cm = d_cm.terrain_detour;
    slope_map = double(cm.slope_deg);
    cost_map = double(cm.cost);
    res_m = double(cm.resolution_m);
    x_min = double(cm.x_range_m(1));
    x_max = double(cm.x_range_m(2));
    y_min = double(cm.y_range_m(1));
    y_max = double(cm.y_range_m(2));
    [rows, cols] = size(cost_map);

    % Sample along Direct Path
    [slope_direct, cost_direct] = sample_map_along_path(w_direct, slope_map, cost_map, x_min, x_max, y_min, y_max, rows, cols);
    max_slope_direct = max(slope_direct);
    mean_cost_direct = mean(cost_direct);

    % Sample along A* Detour Path
    [slope_astar, cost_astar] = sample_map_along_path(w_astar, slope_map, cost_map, x_min, x_max, y_min, y_max, rows, cols);
    max_slope_astar = max(slope_astar);
    mean_cost_astar = mean(cost_astar);

    detour_ratio = len_astar / len_direct;
    max_lateral_detour = max(abs(w_astar(:, 2)));

    % Verification criterion: Direct path violates lethal slope (>= 28 deg), while A* detour stays below 28 deg
    direct_hazardous = (max_slope_direct >= 28.0);
    astar_safe = (max_slope_astar < 28.0);

    res = struct();
    res.direct_length_m = len_direct;
    res.direct_max_slope_deg = max_slope_direct;
    res.direct_mean_cost = mean_cost_direct;
    res.astar_length_m = len_astar;
    res.astar_max_slope_deg = max_slope_astar;
    res.astar_mean_cost = mean_cost_astar;
    res.detour_ratio = detour_ratio;
    res.max_lateral_detour_m = max_lateral_detour;
    res.is_valid = direct_hazardous && astar_safe;

    fprintf('  Direct Route Length     : %.2f m | Max Slope: %.1f deg (LETHAL >= 28 deg)\n', len_direct, max_slope_direct);
    fprintf('  A* Detour Length        : %.2f m | Max Slope: %.1f deg (SAFE < 28 deg)\n', len_astar, max_slope_astar);
    fprintf('  Detour Distance Ratio   : %.2fx\n', detour_ratio);
    fprintf('  Max Lateral Displacement: %.2f m\n', max_lateral_detour);

    if res.is_valid
        fprintf('  Detour Validation Status: PASS (A* Successfully Avoided Hazardous Terrain)\n');
    else
        fprintf('  Detour Validation Status: FAIL\n');
    end
end


function [slopes, costs] = sample_map_along_path(waypoints, slope_map, cost_map, x_min, x_max, y_min, y_max, rows, cols)
    N = size(waypoints, 1);
    slopes = zeros(N, 1);
    costs = zeros(N, 1);
    for i = 1:N
        x = waypoints(i, 1);
        y = waypoints(i, 2);
        % Convert (x, y) to grid row, col
        r = round((x_max - x) / ((x_max - x_min) / (rows - 1))) + 1;
        c = round((y_max - y) / ((y_max - y_min) / (cols - 1))) + 1;
        r = max(1, min(rows, r));
        c = max(1, min(cols, c));
        slopes(i) = slope_map(r, c);
        costs(i) = cost_map(r, c);
    end
end

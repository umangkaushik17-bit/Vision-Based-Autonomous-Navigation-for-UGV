function res = cross_validate_costmap(cm_struct)
% CROSS_VALIDATE_COSTMAP Validates layers, dimensions, and numerical ranges of BEV costmaps.

    fprintf('--- Cross-Validating BEVTerrainCostmap Data Structures ---\n');

    if nargin < 1 || isempty(cm_struct)
        data_path = 'matlab/data/costmap_scenarios.mat';
        if exist(data_path, 'file')
            d = load(data_path);
            cm_struct = d.terrain_detour; % Default to detour scenario
        else
            error('Costmap scenario data not found at %s', data_path);
        end
    end

    cost = double(cm_struct.cost);
    trav = double(cm_struct.traversability);
    obs = logical(cm_struct.obstacle);
    haz = double(cm_struct.hazard);
    unk = logical(cm_struct.unknown);
    res_m = double(cm_struct.resolution_m);
    grid_shape = size(cost);

    % Numerical range checks
    cost_in_range = all(cost(:) >= 0.0 & cost(:) <= 1.0001);
    trav_in_range = all(trav(:) >= 0.0 & trav(:) <= 1.0001);
    haz_in_range  = all(haz(:) >= 0.0 & haz(:) <= 1.0001);
    shape_correct = (grid_shape(1) == 58 && grid_shape(2) == 60);
    res_correct   = (abs(res_m - 0.10) < 1e-4);

    % Lethal cost alignment check: any obstacle cell must have cost >= 0.85
    obs_indices = find(obs);
    lethal_aligned = true;
    if ~isempty(obs_indices)
        lethal_aligned = all(cost(obs_indices) >= 0.85);
    end

    res = struct();
    res.grid_shape = grid_shape;
    res.resolution_m = res_m;
    res.cost_in_range = cost_in_range;
    res.trav_in_range = trav_in_range;
    res.haz_in_range = haz_in_range;
    res.shape_correct = shape_correct;
    res.lethal_aligned = lethal_aligned;
    res.is_valid = cost_in_range && trav_in_range && shape_correct && res_correct && lethal_aligned;

    fprintf('  Grid Dimensions   : %d rows x %d cols (58 x 60 expected)\n', grid_shape(1), grid_shape(2));
    fprintf('  Resolution        : %.2f m (10 cm expected)\n', res_m);
    fprintf('  Cost Range [0, 1] : %s (Min=%.2f, Max=%.2f)\n', mat2str(cost_in_range), min(cost(:)), max(cost(:)));
    fprintf('  Trav Range [0, 1] : %s\n', mat2str(trav_in_range));
    fprintf('  Obstacle Lethal   : %s (All obstacles have cost >= 0.85)\n', mat2str(lethal_aligned));

    if res.is_valid
        fprintf('  Costmap Status    : PASS (Canonical Structure & Range Verified)\n');
    else
        fprintf('  Costmap Status    : FAIL\n');
    end
end

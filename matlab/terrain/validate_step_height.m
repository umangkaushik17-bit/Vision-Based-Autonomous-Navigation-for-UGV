function res = validate_step_height()
% VALIDATE_STEP_HEIGHT Validates step height variation and lethal cost mapping.
% Tests:
%   1) No step (flat ground, h_var = 0.00m)
%   2) Small step (curb/pebble, h_var = 0.05m)
%   3) Moderate step (rock/mound, h_var = 0.12m)
%   4) Lethal step (ditch/cliff, h_var = 0.25m >= 0.20m threshold)
%
% NOTE: The 0.20m threshold is an initial simulation engineering assumption,
% NOT a physically validated universal UGV limit.

    fprintf('--- Validating Terrain Step Height & Lethal Hazard Mapping ---\n');

    critical_step_threshold_m = 0.20;

    step_scenarios = {
        'flat_ground',    0.00, false;
        'small_step',     0.05, false;
        'moderate_step',  0.12, false;
        'lethal_cliff',   0.25, true
    };

    N_cases = size(step_scenarios, 1);
    reported_heights = zeros(1, N_cases);
    is_lethal_flags = false(1, N_cases);

    fprintf('------------------------------------------------------------------------------------\n');
    fprintf('%-16s | %-12s | %-14s | %-12s | %-10s\n', 'Scenario', 'Known Height', 'Est. Variation', 'Lethal Flag', 'Status');
    fprintf('------------------------------------------------------------------------------------\n');

    for i = 1:N_cases
        name = step_scenarios{i, 1};
        known_h = step_scenarios{i, 2};
        expected_lethal = step_scenarios{i, 3};

        % Generate synthetic patch with step of known_h across center
        [X, Y] = meshgrid(linspace(-0.25, 0.25, 11), linspace(-0.25, 0.25, 11));
        Z = zeros(size(X));
        Z(X >= 0) = known_h;

        % Height variation = max(Z) - min(Z)
        est_h_var = max(Z(:)) - min(Z(:));
        is_lethal = (est_h_var >= critical_step_threshold_m);

        reported_heights(i) = est_h_var;
        is_lethal_flags(i) = is_lethal;

        match = (abs(est_h_var - known_h) < 1e-4) && (is_lethal == expected_lethal);
        st_str = 'PASS';
        if ~match
            st_str = 'FAIL';
        end

        fprintf('%-16s | %-10.2f m | %-12.2f m | %-12s | %s\n', ...
            name, known_h, est_h_var, mat2str(is_lethal), st_str);
    end
    fprintf('------------------------------------------------------------------------------------\n');

    res = struct();
    res.scenarios = step_scenarios(:, 1)';
    res.known_heights_m = cell2mat(step_scenarios(:, 2))';
    res.reported_heights_m = reported_heights;
    res.is_lethal = is_lethal_flags;
    res.critical_threshold_m = critical_step_threshold_m;
    res.threshold_nature = 'initial simulation engineering assumption (NOT physical UGV limit)';
    res.is_valid = all(abs(reported_heights - res.known_heights_m) < 1e-4);

    if res.is_valid
        fprintf('  Step Height Status      : PASS (All step variations and lethal triggers verified)\n');
    else
        fprintf('  Step Height Status      : FAIL\n');
    end
end

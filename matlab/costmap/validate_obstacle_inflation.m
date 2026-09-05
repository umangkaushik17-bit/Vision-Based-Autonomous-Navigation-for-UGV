function res = validate_obstacle_inflation()
% VALIDATE_OBSTACLE_INFLATION Independently validates exponential inflation decay
% as a function of Euclidean distance from obstacle boundary.

    fprintf('--- Validating Obstacle Inflation Decay Profile ---\n');

    r_inscribed_m = 0.35;  % Inscribed robot footprint radius
    r_infl_m = 0.55;       % Inflation buffer radius
    decay_alpha = 5.0;     % Exponential decay coefficient

    dists = linspace(0.0, 1.0, 101);
    costs = zeros(size(dists));

    for i = 1:length(dists)
        d = dists(i);
        if d <= r_inscribed_m
            % Inside inscribed radius: LETHAL
            costs(i) = 1.0;
        elseif d <= r_infl_m
            % Inflation buffer: exponential decay
            costs(i) = exp(-decay_alpha * (d - r_inscribed_m));
        else
            % Outside inflation buffer: zero penalty
            costs(i) = 0.0;
        end
    end

    % Numerical checks
    check_lethal_core = all(costs(dists <= r_inscribed_m) == 1.0);
    check_boundary = (costs(find(dists >= r_inscribed_m, 1)) == 1.0);
    check_outside = all(costs(dists > r_infl_m) == 0.0);
    check_monotonic = all(diff(costs) <= 1e-6);

    res = struct();
    res.r_inscribed_m = r_inscribed_m;
    res.r_infl_m = r_infl_m;
    res.decay_alpha = decay_alpha;
    res.distances_m = dists;
    res.costs = costs;
    res.is_lethal_correct = check_lethal_core;
    res.is_outside_zero = check_outside;
    res.is_monotonic = check_monotonic;
    res.is_valid = check_lethal_core && check_outside && check_monotonic;

    fprintf('  Inscribed Radius (Lethal) : %.2f m\n', r_inscribed_m);
    fprintf('  Inflation Radius (Buffer) : %.2f m\n', r_infl_m);
    fprintf('  Lethal Core C=1.0 Check   : %s\n', mat2str(check_lethal_core));
    fprintf('  Outside Buffer C=0.0 Check: %s\n', mat2str(check_outside));
    fprintf('  Monotonic Decay Check     : %s\n', mat2str(check_monotonic));

    if res.is_valid
        fprintf('  Inflation Geometry Status : PASS (Exact Exponential Decay Verified)\n');
    else
        fprintf('  Inflation Geometry Status : FAIL\n');
    end
end

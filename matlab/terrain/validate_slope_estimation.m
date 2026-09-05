function res = validate_slope_estimation()
% VALIDATE_SLOPE_ESTIMATION Evaluates ground slope estimation across known inclinations:
% 0 deg, 5 deg, 10 deg, 15 deg, 20 deg, 25 deg, 30 deg.

    fprintf('--- Validating Terrain Slope Estimation (0 deg to 30 deg) ---\n');

    known_slopes_deg = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0];
    N_slopes = length(known_slopes_deg);

    [X, Y] = meshgrid(linspace(-0.25, 0.25, 7), linspace(-0.25, 0.25, 7));
    est_slopes_deg = zeros(1, N_slopes);
    slope_errors_deg = zeros(1, N_slopes);

    fprintf('------------------------------------------------------------\n');
    fprintf('%-12s | %-15s | %-12s | %-10s\n', 'Known Slope', 'Estimated Slope', 'Abs Error', 'Status');
    fprintf('------------------------------------------------------------\n');

    for i = 1:N_slopes
        theta = known_slopes_deg(i);
        % Plane equation: Z = X * tan(theta)
        Z = X * tand(theta);
        P = [X(:), Y(:), Z(:)];

        % SVD plane normal
        centroid = mean(P, 1);
        [~, ~, V] = svd(P - centroid, 'econ');
        n = V(:, 3);
        n = n / norm(n);

        % Slope angle from vertical (Z-up)
        % theta = arccos(|n_z|)
        est_theta = acosd(min(abs(n(3)), 1.0));
        err = abs(est_theta - theta);

        est_slopes_deg(i) = est_theta;
        slope_errors_deg(i) = err;

        status_str = 'PASS';
        if err > 1e-3
            status_str = 'FAIL';
        end
        fprintf('%-10.1f deg | %-13.4f deg | %-10.2e deg | %s\n', theta, est_theta, err, status_str);
    end
    fprintf('------------------------------------------------------------\n');

    max_err = max(slope_errors_deg);
    mean_err = mean(slope_errors_deg);

    res = struct();
    res.known_slopes_deg = known_slopes_deg;
    res.estimated_slopes_deg = est_slopes_deg;
    res.errors_deg = slope_errors_deg;
    res.max_error_deg = max_err;
    res.mean_error_deg = mean_err;
    res.is_valid = (max_err < 1e-3);

    if res.is_valid
        fprintf('  Slope Estimation Status : PASS (All inclinations accurately recovered)\n');
    else
        fprintf('  Slope Estimation Status : FAIL\n');
    end
end

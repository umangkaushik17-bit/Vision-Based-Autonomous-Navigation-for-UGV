function res = compute_bev_transform_error()
% COMPUTE_BEV_TRANSFORM_ERROR Evaluates metric error between ground truth
% 3D ground coordinates and reconstructed BEV coordinates from synthetic camera rays.
%
% NOTE: Labeled strictly as SYNTHETIC GEOMETRIC VALIDATION.

    fprintf('--- Computing Synthetic BEV Transform Metric Error ---\n');

    calib_int = validate_intrinsics();
    calib_ext = validate_extrinsics();

    K = calib_int.K;
    R = calib_ext.R;
    t = calib_ext.t;
    c_rob = [calib_ext.forward_offset_m; 0.0; calib_ext.height_m];

    % Synthetic ground truth grid on Z=0
    x_gt_vals = linspace(1.0, 5.5, 10);
    y_gt_vals = linspace(-2.5, 2.5, 11);
    [X_gt, Y_gt] = meshgrid(x_gt_vals, y_gt_vals);
    N = numel(X_gt);

    P_gt_all = [X_gt(:), Y_gt(:), zeros(N, 1)]'; % (3, N)

    % Project to camera pixels
    P_cam = R * P_gt_all + t;
    p_proj = K * (P_cam ./ P_cam(3, :));
    u = p_proj(1, :)';
    v = p_proj(2, :)';

    % Filter points that fall within image frame bounds (640x480)
    in_bounds = (u >= 0) & (u < 640) & (v >= 0) & (v < 480);
    u_in = u(in_bounds);
    v_in = v(in_bounds);
    P_gt_in = P_gt_all(:, in_bounds);
    N_in = sum(in_bounds);

    % Inverse ray-casting to reconstruct ground position
    P_reconstructed = zeros(3, N_in);
    K_inv = inv(K);

    for i = 1:N_in
        d_cam = K_inv * [u_in(i); v_in(i); 1.0];
        d_cam = d_cam / norm(d_cam);
        d_rob = R' * d_cam;
        lambda = -c_rob(3) / d_rob(3);
        P_reconstructed(:, i) = c_rob + lambda * d_rob;
    end

    % Metric errors in meters
    diffs = P_reconstructed - P_gt_in;
    errs_m = sqrt(sum(diffs.^2, 1));

    rmse_m = sqrt(mean(errs_m.^2));
    mae_m = mean(errs_m);
    max_m = max(errs_m);

    res = struct();
    res.category = 'Synthetic geometric validation';
    res.total_tested_points = N;
    res.fov_visible_points = N_in;
    res.rmse_m = rmse_m;
    res.mae_m = mae_m;
    res.max_error_m = max_m;
    res.is_valid = (max_m < 1e-4);

    fprintf('  Category               : SYNTHETIC GEOMETRIC VALIDATION\n');
    fprintf('  FOV Visible Test Points: %d / %d\n', N_in, N);
    fprintf('  Reconstruction RMSE    : %.2e meters\n', rmse_m);
    fprintf('  Reconstruction MAE     : %.2e meters\n', mae_m);
    fprintf('  Maximum Error          : %.2e meters\n', max_m);

    if res.is_valid
        fprintf('  Transform Status       : PASS (Exact Reversible Synthetic Geometry)\n');
    else
        fprintf('  Transform Status       : FAIL\n');
    end
end

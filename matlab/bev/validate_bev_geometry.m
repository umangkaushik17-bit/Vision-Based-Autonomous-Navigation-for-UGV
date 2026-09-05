function res = validate_bev_geometry()
% VALIDATE_BEV_GEOMETRY Independently validates ground-plane ray intersection and round-trip BEV reprojection.

    fprintf('--- Validating BEV Mathematical Geometry (Ground Plane Ray Casting) ---\n');

    % Setup camera intrinsics and extrinsics
    calib_int = validate_intrinsics();
    calib_ext = validate_extrinsics();

    K = calib_int.K;
    K_inv = inv(K);
    R = calib_ext.R;
    t = calib_ext.t;
    c_rob = [calib_ext.forward_offset_m; 0.0; calib_ext.height_m];

    % Sample image points below the horizon (ground plane visible area)
    u_samples = linspace(80, 560, 9);
    v_samples = linspace(260, 460, 7);
    [U, V] = meshgrid(u_samples, v_samples);
    pts_img = [U(:), V(:)];
    N = size(pts_img, 1);

    pts_ground_rob = zeros(3, N);
    reproj_pixels = zeros(2, N);
    pixel_errors = zeros(N, 1);

    for i = 1:N
        u = pts_img(i, 1);
        v = pts_img(i, 2);

        % Ray direction in camera optical frame
        d_cam = K_inv * [u; v; 1.0];
        d_cam = d_cam / norm(d_cam);

        % Ray direction in robot frame
        d_rob = R' * d_cam;

        % Intersect with flat ground plane: Z_rob = 0
        % c_rob(3) + lambda * d_rob(3) = 0 => lambda = -c_rob(3) / d_rob(3)
        if d_rob(3) < -1e-4
            lambda = -c_rob(3) / d_rob(3);
            P_ground = c_rob + lambda * d_rob;
            pts_ground_rob(:, i) = P_ground;

            % Round-trip reprojection back into image
            P_cam = R * P_ground + t;
            p_proj = K * (P_cam / P_cam(3));
            u_proj = p_proj(1);
            v_proj = p_proj(2);

            reproj_pixels(:, i) = [u_proj; v_proj];
            pixel_errors(i) = hypot(u_proj - u, v_proj - v);
        else
            pixel_errors(i) = NaN;
        end
    end

    valid_mask = ~isnan(pixel_errors);
    valid_errors = pixel_errors(valid_mask);

    rmse_px = sqrt(mean(valid_errors.^2));
    mae_px = mean(valid_errors);
    max_px = max(valid_errors);

    res = struct();
    res.num_sampled_points = N;
    res.valid_intersections = sum(valid_mask);
    res.rmse_pixels = rmse_px;
    res.mae_pixels = mae_px;
    res.max_error_pixels = max_px;
    res.is_valid = (max_px < 1e-4);

    fprintf('  Sampled Ray Points : %d points\n', N);
    fprintf('  Valid Ground Hits  : %d points\n', sum(valid_mask));
    fprintf('  Round-Trip RMSE    : %.2e px\n', rmse_px);
    fprintf('  Round-Trip MAE     : %.2e px\n', mae_px);
    fprintf('  Round-Trip Max Err : %.2e px\n', max_px);

    if res.is_valid
        fprintf('  BEV Geometry Status: PASS (Exact Mathematical Invertibility Verified)\n');
    else
        fprintf('  BEV Geometry Status: FAIL\n');
    end
end

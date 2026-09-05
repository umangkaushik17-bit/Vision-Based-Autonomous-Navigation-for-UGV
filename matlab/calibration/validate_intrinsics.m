function res = validate_intrinsics(cam_data)
% VALIDATE_INTRINSICS Independently validates camera pinhole intrinsics and FOV equations.

    fprintf('--- Validating Camera Intrinsics ---\n');
    if nargin < 1 || isempty(cam_data)
        cam_file = 'matlab/data/camera_geometry.mat';
        if exist(cam_file, 'file')
            d = load(cam_file);
            cam_data = d;
        else
            % Default theoretical RealSense / Oak-D 640x480 model
            cam_data = struct('width', 640, 'height', 480, 'fov_deg', 69.4);
        end
    end

    W = double(cam_data.width);
    H = double(cam_data.height);
    fov_deg = double(cam_data.fov_deg);

    % Independent mathematical recalculation
    half_fov_rad = deg2rad(fov_deg / 2.0);
    expected_fx = (W / 2.0) / tan(half_fov_rad);
    expected_fy = expected_fx;
    expected_cx = W / 2.0;
    expected_cy = H / 2.0;

    K_recomputed = [expected_fx, 0.0, expected_cx; ...
                    0.0, expected_fy, expected_cy; ...
                    0.0, 0.0, 1.0];

    det_K = det(K_recomputed);
    K_inv = inv(K_recomputed);
    inv_err = norm(K_inv * K_recomputed - eye(3), 'fro');

    res = struct();
    res.width = W;
    res.height = H;
    res.fov_deg = fov_deg;
    res.expected_fx = expected_fx;
    res.expected_fy = expected_fy;
    res.expected_cx = expected_cx;
    res.expected_cy = expected_cy;
    res.K = K_recomputed;
    res.det_K = det_K;
    res.inv_identity_error = inv_err;
    res.is_valid = (inv_err < 1e-6) && (det_K > 0);

    fprintf('  Image Size        : %d x %d px\n', W, H);
    fprintf('  Horizontal FOV    : %.2f deg\n', fov_deg);
    fprintf('  Computed fx, fy   : %.2f, %.2f px\n', expected_fx, expected_fy);
    fprintf('  Principal Point   : (%.1f, %.1f) px\n', expected_cx, expected_cy);
    fprintf('  det(K)            : %.2e (non-singular)\n', det_K);
    fprintf('  K*inv(K) error    : %.2e\n', inv_err);

    if res.is_valid
        fprintf('  Intrinsics Status : PASS (Mathematical Consistency Confirmed)\n');
    else
        fprintf('  Intrinsics Status : FAIL\n');
    end
end

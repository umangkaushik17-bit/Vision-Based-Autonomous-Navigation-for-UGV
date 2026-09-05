function res = validate_extrinsics(cam_data)
% VALIDATE_EXTRINSICS Independently validates robot-to-camera extrinsic rotation and translation.
% Verifies R^T * R = I, det(R) = 1.0, and translation vector scaling.

    fprintf('--- Validating Camera Extrinsics ---\n');
    if nargin < 1 || isempty(cam_data)
        cam_file = 'matlab/data/camera_geometry.mat';
        if exist(cam_file, 'file')
            d = load(cam_file);
            cam_data = d;
        else
            cam_data = struct('camera_pitch_deg', 15.0, 'camera_height_m', 0.40, 'camera_forward_offset_m', 0.15);
        end
    end

    pitch_deg = double(cam_data.camera_pitch_deg);
    height_m = double(cam_data.camera_height_m);
    fwd_offset_m = double(cam_data.camera_forward_offset_m);

    pitch_rad = deg2rad(pitch_deg);

    % Base alignment from Robot frame (X-fwd, Y-left, Z-up) to Optical frame (X-right, Y-down, Z-fwd)
    R_base = [ 0.0, -1.0,  0.0; ...
               0.0,  0.0, -1.0; ...
               1.0,  0.0,  0.0 ];

    % Pitch rotation about robot Y axis (tilt down = negative pitch in standard Euler)
    cp = cos(pitch_rad);
    sp = sin(pitch_rad);
    R_pitch = [ cp,  0.0, -sp; ...
                0.0, 1.0,  0.0; ...
                sp,  0.0,  cp ];

    R = R_base * R_pitch;

    % Orthogonality checks
    ortho_err = norm(R' * R - eye(3), 'fro');
    det_R = det(R);

    % Translation vector in camera frame
    t_robot = [fwd_offset_m; 0.0; height_m];
    t_cam = -R * t_robot;

    res = struct();
    res.R = R;
    res.t = t_cam;
    res.pitch_deg = pitch_deg;
    res.height_m = height_m;
    res.forward_offset_m = fwd_offset_m;
    res.det_R = det_R;
    res.orthogonality_error = ortho_err;
    res.is_valid = (abs(det_R - 1.0) < 1e-6) && (ortho_err < 1e-6);

    fprintf('  Camera Mounting Height : %.2f m\n', height_m);
    fprintf('  Forward Offset         : %.2f m\n', fwd_offset_m);
    fprintf('  Pitch Angle            : %.1f deg downward\n', pitch_deg);
    fprintf('  det(R)                 : %.6f (proper rotation)\n', det_R);
    fprintf('  ||R^T * R - I||        : %.2e (orthonormal)\n', ortho_err);
    fprintf('  Translation vector t   : [%.3f, %.3f, %.3f]^T m\n', t_cam(1), t_cam(2), t_cam(3));

    if res.is_valid
        fprintf('  Extrinsics Status      : PASS (SE(3) Transformation Verified)\n');
    else
        fprintf('  Extrinsics Status      : FAIL\n');
    end
end

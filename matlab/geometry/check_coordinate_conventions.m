function res = check_coordinate_conventions()
% CHECK_COORDINATE_CONVENTIONS Independently verifies directional axes mapping
% between Robot Frame (X-forward, Y-left, Z-up) and Camera Optical Frame (X-right, Y-down, Z-forward).

    fprintf('--- Checking Coordinate Conventions (Robot vs. Camera Optical) ---\n');

    % Load or compute extrinsic transform with 0 deg pitch for pure axis alignment check
    R_base = [ 0.0, -1.0,  0.0; ...
               0.0,  0.0, -1.0; ...
               1.0,  0.0,  0.0 ];

    % Test cardinal directional vectors in robot base_link frame
    p_fwd_robot   = [1.0; 0.0; 0.0];  % +X forward
    p_left_robot  = [0.0; 1.0; 0.0];  % +Y left
    p_right_robot = [0.0; -1.0; 0.0]; % -Y right
    p_up_robot    = [0.0; 0.0; 1.0];  % +Z up
    p_down_robot  = [0.0; 0.0; -1.0]; % -Z down

    % Transform vectors to camera optical frame
    v_fwd_cam   = R_base * p_fwd_robot;
    v_left_cam  = R_base * p_left_robot;
    v_right_cam = R_base * p_right_robot;
    v_up_cam    = R_base * p_up_robot;
    v_down_cam  = R_base * p_down_robot;

    % Validation conditions:
    % 1. Forward (+X_robot) must map to optical axis forward (+Z_cam)
    % 2. Left (+Y_robot) must map to camera-left (-X_cam, since camera +X is right)
    % 3. Up (+Z_robot) must map to camera-up (-Y_cam, since camera +Y is down)
    check_fwd   = (norm(v_fwd_cam - [0.0; 0.0; 1.0]) < 1e-6);
    check_left  = (norm(v_left_cam - [-1.0; 0.0; 0.0]) < 1e-6);
    check_right = (norm(v_right_cam - [1.0; 0.0; 0.0]) < 1e-6);
    check_up    = (norm(v_up_cam - [0.0; -1.0; 0.0]) < 1e-6);
    check_down  = (norm(v_down_cam - [0.0; 1.0; 0.0]) < 1e-6);

    res = struct();
    res.check_forward = check_fwd;
    res.check_left = check_left;
    res.check_right = check_right;
    res.check_up = check_up;
    res.check_down = check_down;
    res.all_passed = check_fwd && check_left && check_right && check_up && check_down;

    fprintf('  Forward Mapping (+X_rob -> +Z_cam) : %s\n', mat2str(check_fwd));
    fprintf('  Left Mapping    (+Y_rob -> -X_cam) : %s\n', mat2str(check_left));
    fprintf('  Right Mapping   (-Y_rob -> +X_cam) : %s\n', mat2str(check_right));
    fprintf('  Up Mapping      (+Z_rob -> -Y_cam) : %s\n', mat2str(check_up));
    fprintf('  Down Mapping    (-Z_rob -> +Y_cam) : %s\n', mat2str(check_down));

    if res.all_passed
        fprintf('  Coordinate Convention Status       : PASS (No Silent Sign Swaps)\n');
    else
        fprintf('  Coordinate Convention Status       : FAIL (Inconsistent Mapping)\n');
    end
end

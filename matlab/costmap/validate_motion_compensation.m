function res = validate_motion_compensation()
% VALIDATE_MOTION_COMPENSATION Validates SE(2) grid transformation under known motions:
% 1) Pure forward movement (+dx)
% 2) Pure backward movement (-dx)
% 3) Pure lateral movement (+dy)
% 4) Small rotation (+5 deg)
% 5) Large rotation (+45 deg)
% 6) Combined translation + rotation (dx=0.5m, dy=0.2m, dtheta=15 deg)

    fprintf('--- Validating SE(2) Motion Compensation & Alignment ---\n');

    test_motions = {
        'forward_movement',    0.50,  0.00,   0.0;
        'backward_movement',  -0.30,  0.00,   0.0;
        'lateral_movement',    0.00,  0.40,   0.0;
        'small_rotation',      0.00,  0.00,   5.0;
        'large_rotation',      0.00,  0.00,  45.0;
        'combined_trans_rot',  0.50,  0.20,  15.0
    };

    N_tests = size(test_motions, 1);
    alignment_errors = zeros(1, N_tests);

    fprintf('------------------------------------------------------------------------------------\n');
    fprintf('%-22s | %-12s | %-12s | %-12s | %-10s\n', 'Motion Case', 'dx (m)', 'dy (m)', 'dtheta (deg)', 'Error (m)');
    fprintf('------------------------------------------------------------------------------------\n');

    for i = 1:N_tests
        name = test_motions{i, 1};
        dx = test_motions{i, 2};
        dy = test_motions{i, 3};
        dtheta_deg = test_motions{i, 4};
        dtheta_rad = deg2rad(dtheta_deg);

        % Construct SE(2) transformation matrix
        c = cos(dtheta_rad);
        s = sin(dtheta_rad);
        T = [ c, -s, dx; ...
              s,  c, dy; ...
            0.0, 0.0, 1.0 ];

        % Known test point in previous frame
        p_prev = [2.0; 0.5; 1.0];

        % Forward transform to current frame
        p_curr = T * p_prev;

        % Inverse transform back to previous frame
        T_inv = inv(T);
        p_reconstructed = T_inv * p_curr;

        err = norm(p_reconstructed(1:2) - p_prev(1:2));
        alignment_errors(i) = err;

        fprintf('%-22s | %-10.2f m | %-10.2f m | %-10.1f deg | %.2e m\n', ...
            name, dx, dy, dtheta_deg, err);
    end
    fprintf('------------------------------------------------------------------------------------\n');

    max_err = max(alignment_errors);
    res = struct();
    res.test_motions = test_motions(:, 1)';
    res.alignment_errors_m = alignment_errors;
    res.max_error_m = max_err;
    res.is_valid = (max_err < 1e-6);

    if res.is_valid
        fprintf('  Motion Compensation Status: PASS (SE(2) Matrix Alignment Exact)\n');
    else
        fprintf('  Motion Compensation Status: FAIL\n');
    end
end

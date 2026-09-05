function res = validate_phase_9_transforms()
% VALIDATE_PHASE_9_TRANSFORMS Independently validates Phase 9 SE(3) coordinate
% transforms, camera extrinsics, and world mapping parity using MATLAB.

    fprintf('==============================================================================\n');
    fprintf('         SIH26126 PHASE 9: MATLAB SPATIAL TRANSFORMS VALIDATION\n');
    fprintf('==============================================================================\n\n');

    script_dir = fileparts(mfilename('fullpath'));
    root_dir = fullfile(script_dir, '..');
    addpath(genpath(root_dir));

    mat_file = fullfile(root_dir, 'data', 'phase_9_transforms.mat');
    if ~exist(mat_file, 'file')
        error('Phase 9 transforms data not found at: %s', mat_file);
    end

    d = load(mat_file);
    T_body_cam = double(d.T_body_cam);
    T_world_body = double(d.T_world_body);
    pts_cam = double(d.pts_cam);
    pts_body_py = double(d.pts_body);
    pts_world_py = double(d.pts_world);

    % 1. Compute Body Coordinates in MATLAB:
    % P_body = P_cam @ R_bc^T + t_bc^T
    R_bc = T_body_cam(1:3, 1:3);
    t_bc = T_body_cam(1:3, 4)';
    pts_body_matlab = pts_cam * R_bc' + t_bc;

    diff_body = max(abs(pts_body_matlab(:) - pts_body_py(:)));
    fprintf('  Camera -> Body Parity Error : %.4e m\n', diff_body);

    % 2. Compute World Coordinates in MATLAB:
    % P_world = P_body @ R_wb^T + t_wb^T
    R_wb = T_world_body(1:3, 1:3);
    t_wb = T_world_body(1:3, 4)';
    pts_world_matlab = pts_body_matlab * R_wb' + t_wb;

    diff_world = max(abs(pts_world_matlab(:) - pts_world_py(:)));
    fprintf('  Body -> World Parity Error   : %.4e m\n', diff_world);

    % 3. Check Full Chain T_wc = T_wb * T_bc:
    T_world_cam = T_world_body * T_body_cam;
    R_wc = T_world_cam(1:3, 1:3);
    t_wc = T_world_cam(1:3, 4)';
    pts_direct = pts_cam * R_wc' + t_wc;
    diff_chain = max(abs(pts_direct(:) - pts_world_py(:)));
    fprintf('  Full Chain Direct Parity    : %.4e m\n', diff_chain);

    % 4. Verify Inverse SE(3): T * inv(T) == Eye(4)
    T_inv_wb = [R_wb', -R_wb' * t_wb'; 0, 0, 0, 1];
    diff_inv = max(abs(reshape(T_world_body * T_inv_wb - eye(4), [], 1)));
    fprintf('  SE(3) Inverse Identity Error: %.4e\n', diff_inv);

    is_valid = (diff_body < 1e-10) && (diff_world < 1e-10) && (diff_chain < 1e-10) && (diff_inv < 1e-10);

    res = struct();
    res.diff_body_max_m = diff_body;
    res.diff_world_max_m = diff_world;
    res.diff_chain_max_m = diff_chain;
    res.diff_inv_max = diff_inv;
    res.is_valid = is_valid;

    fprintf('\n  Numerical Parity Threshold  : 1.0000e-10\n');
    if is_valid
        fprintf('  Validation Verdict          : [PASS] Strict Numerical Parity Confirmed\n\n');
    else
        fprintf('  Validation Verdict          : [FAIL] Parity Exceeded Threshold\n\n');
    end

    report_file = fullfile(root_dir, 'results', 'reports', 'phase_9_transforms_report.mat');
    save(report_file, 'res');
end

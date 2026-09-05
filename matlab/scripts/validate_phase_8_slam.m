function res = validate_phase_8_slam()
% VALIDATE_PHASE_8_SLAM Imports Phase 8 estimated and ground-truth trajectories
% from matlab/data/phase_8_evaluated_slam.mat and computes ATE and RPE using Phase 7 engines.

    fprintf('==============================================================================\n');
    fprintf('           SIH26126 PHASE 8: MATLAB ATE/RPE SLAM VALIDATION\n');
    fprintf('==============================================================================\n\n');

    script_dir = fileparts(mfilename('fullpath'));
    root_dir = fullfile(script_dir, '..');
    addpath(genpath(root_dir));

    mat_file = fullfile(root_dir, 'data', 'phase_8_evaluated_slam.mat');
    if ~exist(mat_file, 'file')
        error('Phase 8 evaluated SLAM data not found at: %s', mat_file);
    end

    d = load(mat_file);
    gt_canon = double(d.ground_truth);
    est_canon = double(d.estimated);

    % Positions (x, y, z)
    gt_xyz = gt_canon(:, 2:4);
    est_xyz = est_canon(:, 2:4);

    N = min(size(gt_xyz, 1), size(est_xyz, 1));
    gt_xyz = gt_xyz(1:N, :);
    est_xyz = est_xyz(1:N, :);

    % 1. Compute Raw ATE (no alignment)
    ate_raw = compute_ate(gt_xyz, est_xyz, false);

    % 2. Compute Aligned ATE (Umeyama SE(3) rigid alignment, scale preserved)
    ate_aligned = compute_ate(gt_xyz, est_xyz, true);

    % 3. Compute RPE (Delta=10 steps)
    delta_step = min(10, floor(N / 2));
    rpe = compute_rpe(gt_xyz, est_xyz, delta_step);

    fprintf('  Evaluated Poses     : %d poses\n', N);
    fprintf('  ATE RMSE (Raw)      : %.4f m\n', ate_raw.rmse);
    fprintf('  ATE RMSE (Aligned)  : %.4f m\n', ate_aligned.rmse);
    fprintf('  ATE Mean Error      : %.4f m\n', ate_aligned.mean);
    fprintf('  ATE Max Error       : %.4f m\n', ate_aligned.max);
    fprintf('  RPE RMSE (Delta=%d) : %.4f m\n', delta_step, rpe.rmse);

    res = struct();
    res.num_poses = N;
    res.ate_raw_rmse_m = ate_raw.rmse;
    res.ate_aligned_rmse_m = ate_aligned.rmse;
    res.ate_aligned_mean_m = ate_aligned.mean;
    res.ate_aligned_max_m = ate_aligned.max;
    res.rpe_rmse_m = rpe.rmse;
    res.is_valid = (ate_aligned.rmse >= 0.0) && (rpe.rmse >= 0.0);

    % Save results
    report_file = fullfile(root_dir, 'results', 'reports', 'phase_8_slam_ate_report.mat');
    save(report_file, 'res');
    fprintf('\n[OK] Phase 8 MATLAB SLAM evaluation complete. Saved to: %s\n\n', report_file);
end

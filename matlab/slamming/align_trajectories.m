function [traj_est_aligned, R_align, t_align] = align_trajectories(traj_gt, traj_est)
% ALIGN_TRAJECTORIES Aligns estimated trajectory to ground truth using rigid Umeyama SE(2)/SE(3) alignment.
% Preserves metric scale (scale s = 1.0) for metric visual odometry / SLAM.
% Inputs:
%   traj_gt:  (N, 2) or (N, 3) reference trajectory positions
%   traj_est: (N, 2) or (N, 3) estimated trajectory positions
% Outputs:
%   traj_est_aligned: (N, 2) or (N, 3) aligned estimated trajectory
%   R_align: (2, 2) or (3, 3) rotation matrix
%   t_align: (2, 1) or (3, 1) translation vector

    dim = size(traj_gt, 2);
    N = size(traj_gt, 1);

    if size(traj_est, 1) ~= N || size(traj_est, 2) ~= dim
        error('Trajectories must have identical dimensions for alignment');
    end

    % Centroids
    mu_gt = mean(traj_gt, 1)';
    mu_est = mean(traj_est, 1)';

    % Center the points
    X = traj_gt' - mu_gt;   % (dim, N)
    Y = traj_est' - mu_est; % (dim, N)

    % Covariance matrix
    H = Y * X' / N;

    % SVD of covariance
    [U, ~, V] = svd(H);

    % Rotation matrix
    d = det(V * U');
    S = eye(dim);
    if d < 0
        S(end, end) = -1;
    end
    R_align = V * S * U';

    % Translation vector
    t_align = mu_gt - R_align * mu_est;

    % Apply rigid alignment
    traj_est_aligned = (R_align * traj_est' + t_align)';
end

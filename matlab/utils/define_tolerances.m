function tol = define_tolerances()
% DEFINE_TOLERANCES Numerical comparison thresholds for cross-language validation.

    tol = struct();
    % Geometric and linear tolerances
    tol.distance_m = 1e-3;          % 1 millimeter positional tolerance
    tol.bev_coord_m = 1e-3;         % 1 millimeter BEV grid coordinates
    tol.resolution_m = 1e-4;        % 0.1 millimeter grid resolution

    % Angular and rotation tolerances
    tol.angle_deg = 1e-2;           % 0.01 degree angular tolerance
    tol.angle_rad = 1e-4;           % 0.1 milliradian angular tolerance
    tol.matrix_ortho = 1e-5;        % R^T * R - I max residual
    tol.matrix_det = 1e-5;          % det(R) - 1.0 tolerance

    % Float parity tolerances between Python and MATLAB
    tol.float_relative = 1e-3;      % 0.1% relative difference
    tol.float_absolute = 1e-4;      % 1e-4 absolute difference

    % Trajectory tracking tolerances
    tol.ate_m = 1e-3;               % 1 millimeter ATE accuracy
    tol.rpe_trans_m = 1e-3;         % 1 millimeter RPE translational drift
    tol.rpe_rot_rad = 1e-4;         % 0.1 milliradian RPE rotational drift
end

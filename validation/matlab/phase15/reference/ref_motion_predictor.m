function [waypoints, P_trajectories, r_uncertainties] = ref_motion_predictor(x0, P0, time_offsets, q, sigma_scale)
% REF_MOTION_PREDICTOR Independent CV motion predictor in MATLAB.
%
% Inputs:
%   x0            - 4x1 initial state [px; py; vx; vy]
%   P0            - 4x4 initial covariance matrix
%   time_offsets  - 1xK vector of future time offsets (seconds)
%   q             - Continuous process noise acceleration spectral density (default 1.0)
%   sigma_scale   - Confidence multiplier for uncertainty radius (default 3.0)
%
% Outputs:
%   waypoints       - Kx4 matrix of predicted states [px, py, vx, vy]
%   P_trajectories  - 4x4xK array of predicted covariances
%   r_uncertainties - 1xK vector of uncertainty radii (meters)

    if nargin < 4 || isempty(q), q = 1.0; end
    if nargin < 5 || isempty(sigma_scale), sigma_scale = 3.0; end

    K = length(time_offsets);
    waypoints = zeros(K, 4);
    P_trajectories = zeros(4, 4, K);
    r_uncertainties = zeros(1, K);

    x0_col = x0(:);

    for i = 1:K
        tau = time_offsets(i);
        if tau < 1e-9
            x_pred = x0_col;
            P_future = P0;
        else
            % State transition matrix F(tau)
            F = [1.0, 0.0, tau, 0.0;
                 0.0, 1.0, 0.0, tau;
                 0.0, 0.0, 1.0, 0.0;
                 0.0, 0.0, 0.0, 1.0];

            % Process noise Q(tau)
            tau2 = tau * tau;
            tau3 = tau2 * tau;
            Q = q * [tau3 / 3.0, 0.0,        tau2 / 2.0, 0.0;
                     0.0,        tau3 / 3.0, 0.0,        tau2 / 2.0;
                     tau2 / 2.0, 0.0,        tau,        0.0;
                     0.0,        tau2 / 2.0, 0.0,        tau];

            x_pred = F * x0_col;
            P_future = F * P0 * F' + Q;
        end

        P_future = 0.5 * (P_future + P_future'); % Symmetry

        % 2D position sub-covariance
        P_pos = P_future(1:2, 1:2);
        eigvals = eig(P_pos);
        max_eigval = max(0.0, max(eigvals));
        r_unc = sigma_scale * sqrt(max_eigval);

        waypoints(i, :) = x_pred';
        P_trajectories(:, :, i) = P_future;
        r_uncertainties(i) = r_unc;
    end
end

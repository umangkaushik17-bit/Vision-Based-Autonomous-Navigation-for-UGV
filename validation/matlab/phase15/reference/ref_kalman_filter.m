function [x_pred, P_pred, y, S, K, x_upd, P_upd, d2, accepted] = ref_kalman_filter(x_prev, P_prev, dt, z, q, sigma_p, chi2_thresh)
% REF_KALMAN_FILTER Independent 4-state Constant-Velocity Kalman Filter in MATLAB.
%
% State: x = [px; py; vx; vy]
% Measurement: z = [mx; my]
%
% Inputs:
%   x_prev       - 4x1 prior state vector
%   P_prev       - 4x4 prior covariance matrix
%   dt           - Time step (seconds)
%   z            - 2x1 measurement vector (or [] for predict only)
%   q            - Process noise acceleration spectral density (m^2/s^3)
%   sigma_p      - Measurement position noise standard deviation (meters)
%   chi2_thresh  - Chi-square gating threshold (e.g. 9.21 for 2-DOF 99%)
%
% Outputs:
%   x_pred       - 4x1 predicted state
%   P_pred       - 4x4 predicted covariance
%   y            - 2x1 innovation residual (z - H*x_pred)
%   S            - 2x2 innovation covariance
%   K            - 4x2 Kalman gain
%   x_upd        - 4x1 posterior state (if accepted, else x_pred)
%   P_upd        - 4x4 posterior covariance (Joseph form, else P_pred)
%   d2           - Mahalanobis distance squared
%   accepted     - Boolean indicating gating acceptance

    if nargin < 5 || isempty(q), q = 1.0; end
    if nargin < 6 || isempty(sigma_p), sigma_p = 0.05; end
    if nargin < 7 || isempty(chi2_thresh), chi2_thresh = 9.21; end

    % 1. Transition Matrix F(dt)
    F = [1.0, 0.0, dt,  0.0;
         0.0, 1.0, 0.0, dt;
         0.0, 0.0, 1.0, 0.0;
         0.0, 0.0, 0.0, 1.0];

    % 2. Process Noise Matrix Q(dt) - Continuous White-Noise Acceleration
    dt2 = dt * dt;
    dt3 = dt2 * dt;
    Q = q * [dt3 / 3.0, 0.0,       dt2 / 2.0, 0.0;
             0.0,       dt3 / 3.0, 0.0,       dt2 / 2.0;
             dt2 / 2.0, 0.0,       dt,        0.0;
             0.0,       dt2 / 2.0, 0.0,       dt];

    % 3. State Prediction: x(k|k-1) = F * x(k-1|k-1)
    x_pred = F * x_prev;

    % 4. Covariance Prediction: P(k|k-1) = F * P * F^T + Q
    P_pred = F * P_prev * F' + Q;
    P_pred = 0.5 * (P_pred + P_pred'); % Enforce symmetry

    if isempty(z)
        y = []; S = []; K = [];
        x_upd = x_pred; P_upd = P_pred;
        d2 = 0.0; accepted = true;
        return;
    end

    % 5. Measurement Matrix H and Measurement Covariance R
    H = [1.0, 0.0, 0.0, 0.0;
         0.0, 1.0, 0.0, 0.0];
    R = [sigma_p^2, 0.0;
         0.0,       sigma_p^2];

    % 6. Innovation residual: y = z - H * x_pred
    y = z - H * x_pred;

    % 7. Innovation covariance: S = H * P_pred * H^T + R
    S = H * P_pred * H' + R;
    S = 0.5 * (S + S');

    % 8. Mahalanobis distance squared: d^2 = y^T * S^-1 * y
    % Use exact 2x2 analytic inverse or linear solve
    y_col = y(:);
    S_inv_y = S \ y_col;
    d2 = y_col' * S_inv_y;

    % 9. Outlier gating
    if d2 > chi2_thresh
        accepted = false;
        K = (P_pred * H') / S;
        x_upd = x_pred;
        P_upd = P_pred;
        return;
    end

    accepted = true;

    % 10. Kalman Gain: K = P_pred * H^T * S^-1
    K = (P_pred * H') / S;

    % 11. State Update: x(k|k) = x_pred + K * y
    x_upd = x_pred + K * y_col;

    % 12. Joseph-Form Covariance Update:
    % P(k|k) = (I - K*H) * P_pred * (I - K*H)^T + K * R * K^T
    I4 = eye(4);
    I_KH = I4 - K * H;
    P_upd = I_KH * P_pred * I_KH' + K * R * K';
    P_upd = 0.5 * (P_upd + P_upd'); % Enforce symmetry
end

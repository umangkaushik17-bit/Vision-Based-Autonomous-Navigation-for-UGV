function [vx, vy, speed, heading, residual_norm, status] = ref_least_squares_velocity(t_series, x_series, y_series)
% REF_LEAST_SQUARES_VELOCITY Independent OLS linear velocity estimator.
%
% Inputs:
%   t_series - 1xN vector of timestamps (seconds)
%   x_series - 1xN vector of x positions (meters)
%   y_series - 1xN vector of y positions (meters)
%
% Outputs:
%   vx, vy         - Estimated linear velocities (m/s)
%   speed          - Estimated scalar speed hypot(vx, vy) (m/s)
%   heading        - Estimated velocity heading angle atan2(vy, vx) (rad)
%   residual_norm  - Combined OLS residual sum of squares
%   status         - 'OK', 'INSUFFICIENT_HISTORY', or 'DEGENERATE_TIMESTAMPS'

    N = length(t_series);
    if N < 2 || length(x_series) ~= N || length(y_series) ~= N
        vx = 0.0; vy = 0.0; speed = 0.0; heading = 0.0; residual_norm = 0.0;
        status = 'INSUFFICIENT_HISTORY';
        return;
    end

    t_mean = mean(t_series);
    x_mean = mean(x_series);
    y_mean = mean(y_series);

    dt = t_series - t_mean;
    dt_sq_sum = sum(dt .^ 2);

    if dt_sq_sum < 1e-12
        vx = 0.0; vy = 0.0; speed = 0.0; heading = 0.0; residual_norm = 0.0;
        status = 'DEGENERATE_TIMESTAMPS';
        return;
    end

    % Analytical OLS slope formula
    vx = sum(dt .* (x_series - x_mean)) / dt_sq_sum;
    vy = sum(dt .* (y_series - y_mean)) / dt_sq_sum;

    speed = hypot(vx, vy);
    heading = atan2(vy, vx);

    % Compute residuals
    x_fitted = x_mean + vx * dt;
    y_fitted = y_mean + vy * dt;
    res_x = x_series - x_fitted;
    res_y = y_series - y_fitted;
    residual_norm = sqrt(sum(res_x .^ 2) + sum(res_y .^ 2));

    status = 'OK';
end

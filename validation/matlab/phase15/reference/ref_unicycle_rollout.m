function [traj_pts] = ref_unicycle_rollout(x0, y0, yaw0, v, omega, dt, steps)
% REF_UNICYCLE_ROLLOUT Independent kinematic unicycle forward rollout in MATLAB.
%
% Implements the midpoint circular-arc integration used in DWAPlanner:
%   theta_mid = yaw + 0.5 * omega * dt
%   x = x + v * cos(theta_mid) * dt
%   y = y + v * sin(theta_mid) * dt
%   yaw = wrapToPi(yaw + omega * dt)
%
% Inputs:
%   x0, y0, yaw0 - Initial pose (meters, radians)
%   v, omega     - Linear (m/s) and angular (rad/s) velocities
%   dt           - Integration step size (seconds)
%   steps        - Number of forward rollout steps
%
% Outputs:
%   traj_pts     - steps x 4 matrix [x, y, yaw, t]

    traj_pts = zeros(steps, 4);
    curr_x = x0;
    curr_y = y0;
    curr_yaw = yaw0;
    curr_t = 0.0;

    for k = 1:steps
        theta_mid = curr_yaw + 0.5 * omega * dt;
        curr_x = curr_x + v * cos(theta_mid) * dt;
        curr_y = curr_y + v * sin(theta_mid) * dt;
        curr_yaw = curr_yaw + omega * dt;
        % Normalize to [-pi, pi]
        curr_yaw = atan2(sin(curr_yaw), cos(curr_yaw));
        curr_t = curr_t + dt;

        traj_pts(k, :) = [curr_x, curr_y, curr_yaw, curr_t];
    end
end

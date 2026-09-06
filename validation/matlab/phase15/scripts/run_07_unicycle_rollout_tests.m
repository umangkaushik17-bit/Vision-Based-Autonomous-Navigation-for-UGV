% RUN_07_UNICYCLE_ROLLOUT_TESTS
% Validates midpoint circular-arc kinematic unicycle rollout integration against analytical solutions.

clear; clc;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'reference'));
resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

fprintf('=== [Phase 15.07] Running Unicycle Rollout MATLAB Tests ===\n');

% Analytical comparison:
% For constant v and w, the exact trajectory is a circular arc of radius R = v / w.
% Center of circle: (x0 - R*sin(theta0), y0 + R*cos(theta0))
% x(t) = x0 + (v/w)*(sin(theta0 + w*t) - sin(theta0))
% y(t) = y0 - (v/w)*(cos(theta0 + w*t) - cos(theta0))
% theta(t) = theta0 + w*t

x0 = 0.0; y0 = 0.0; theta0 = 0.0;
v = 1.2;
w = 0.4;
dt = 0.05;
steps = 40; % 2.0 seconds horizon

traj = ref_unicycle_rollout(x0, y0, theta0, v, w, dt, steps);

max_pos_err = 0.0;
max_yaw_err = 0.0;

for k = 1:size(traj, 1)
    t = (k - 1) * dt;
    x_exact = x0 + (v / w) * (sin(theta0 + w * t) - sin(theta0));
    y_exact = y0 - (v / w) * (cos(theta0 + w * t) - cos(theta0));
    th_exact = atan2(sin(theta0 + w * t), cos(theta0 + w * t));
    
    pos_err = norm([traj(k, 1) - x_exact, traj(k, 2) - y_exact]);
    yaw_err = abs(atan2(sin(traj(k, 3) - th_exact), cos(traj(k, 3) - th_exact)));
    
    if pos_err > max_pos_err, max_pos_err = pos_err; end
    if yaw_err > max_yaw_err, max_yaw_err = yaw_err; end
end

fprintf('Midpoint vs Exact Arc Error: max_pos_err = %.4e m, max_yaw_err = %.4e rad\n', ...
    max_pos_err, max_yaw_err);

% Midpoint rule is 2nd order accurate: on dt=0.05s, error should be well under 0.01m
assert(max_pos_err < 0.005, 'Position error exceeds 2nd-order kinematic tolerance');
assert(max_yaw_err < 1e-6, 'Yaw error exceeds tolerance');
fprintf('Unicycle Rollout Test: PASS\n\n');

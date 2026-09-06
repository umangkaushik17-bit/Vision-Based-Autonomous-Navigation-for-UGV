function [cost, risk, is_nominal, is_uncertainty] = ref_dynamic_occupancy(...
    x_coords, y_coords, obs_pos, obs_size, obs_yaw, r_unc, sigma_scale)
% REF_DYNAMIC_OCCUPANCY Independent reference evaluation of dynamic obstacle occupancy.
%
% Syntax:
%   [cost, risk, is_nominal, is_uncertainty] = ref_dynamic_occupancy(...
%       x_coords, y_coords, obs_pos, obs_size, obs_yaw, r_unc, sigma_scale)
%
% Inputs:
%   x_coords    - Metric X coordinates of evaluation points (vector or matrix)
%   y_coords    - Metric Y coordinates of evaluation points (vector or matrix)
%   obs_pos     - Obstacle center [x, y] (1x2)
%   obs_size    - Obstacle bounding box [length, width] (1x2)
%   obs_yaw     - Obstacle orientation in radians (scalar)
%   r_unc       - Obstacle positional uncertainty radius (scalar, 3*sigma)
%   sigma_scale - Scaling factor between uncertainty radius and sigma (default: 3.0)
%
% Outputs:
%   cost           - Dynamic cost values in [0.0, 1.0] (same size as x_coords)
%   risk           - Discrete risk level: 0=FREE, 1=LOW, 2=MED, 3=HIGH, 4=LETHAL
%   is_nominal     - Boolean flag for nominal physical footprint
%   is_uncertainty - Boolean flag for uncertainty zone

    if nargin < 7 || isempty(sigma_scale)
        sigma_scale = 3.0;
    end

    half_l = obs_size(1) / 2.0;
    half_w = obs_size(2) / 2.0;

    dx = x_coords - obs_pos(1);
    dy = y_coords - obs_pos(2);

    cos_yaw = cos(obs_yaw);
    sin_yaw = sin(obs_yaw);

    % Rotate into obstacle-centric coordinate frame
    local_x = cos_yaw .* dx + sin_yaw .* dy;
    local_y = -sin_yaw .* dx + cos_yaw .* dy;

    % Distance to nominal bounding box boundary
    d_x = max(0.0, abs(local_x) - half_l);
    d_y = max(0.0, abs(local_y) - half_w);
    dist_to_nominal = sqrt(d_x.^2 + d_y.^2);

    cost = zeros(size(x_coords));
    risk = zeros(size(x_coords));
    is_nominal = (dist_to_nominal <= 1e-5);

    sigma_pos = max(1e-4, r_unc / sigma_scale);
    is_uncertainty = (dist_to_nominal > 1e-5) & (dist_to_nominal <= r_unc);

    % 1. Nominal footprint (strictly lethal)
    cost(is_nominal) = 1.0;
    risk(is_nominal) = 4; % LETHAL

    % 2. High risk zone: (0, 1*sigma] -> cost in [0.70, 0.95)
    mask_high = is_uncertainty & (dist_to_nominal <= sigma_pos);
    if any(mask_high(:))
        norm_d = dist_to_nominal(mask_high) / sigma_pos;
        cost(mask_high) = 0.95 - 0.25 * norm_d;
        risk(mask_high) = 3; % HIGH_RISK
    end

    % 3. Medium risk zone: (1*sigma, 2*sigma] -> cost in [0.40, 0.70)
    mask_med = is_uncertainty & (dist_to_nominal > sigma_pos) & (dist_to_nominal <= 2.0 * sigma_pos);
    if any(mask_med(:))
        norm_d = (dist_to_nominal(mask_med) - sigma_pos) / sigma_pos;
        cost(mask_med) = 0.70 - 0.30 * norm_d;
        risk(mask_med) = 2; % MEDIUM_RISK
    end

    % 4. Low risk zone: (2*sigma, 3*sigma] -> cost in [0.15, 0.40)
    mask_low = is_uncertainty & (dist_to_nominal > 2.0 * sigma_pos) & (dist_to_nominal <= 3.0 * sigma_pos);
    if any(mask_low(:))
        norm_d = (dist_to_nominal(mask_low) - 2.0 * sigma_pos) / sigma_pos;
        cost(mask_low) = 0.40 - 0.25 * norm_d;
        risk(mask_low) = 1; % LOW_RISK
    end
end

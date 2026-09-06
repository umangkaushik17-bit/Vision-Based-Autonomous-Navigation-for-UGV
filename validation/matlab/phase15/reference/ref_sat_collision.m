function [is_collision, min_sep, corners1, corners2] = ref_sat_collision(center1, size1, yaw1, center2, size2, yaw2, margin1, margin2)
% REF_SAT_COLLISION Independent 2D Separating Axis Theorem (SAT) collision evaluator in MATLAB.
%
% Inputs:
%   center1, center2 - [x, y] coordinates of polygon/OBB centers (meters)
%   size1, size2     - [length, width] dimensions (meters)
%   yaw1, yaw2       - Heading angles in radians
%   margin1, margin2 - Safety expansion margins (default 0.0)
%
% Outputs:
%   is_collision     - Boolean: true if polygons overlap, false if separated
%   min_sep          - Minimum separating gap along any axis (<0 if overlap)
%   corners1         - 4x2 matrix of vertices for polygon 1
%   corners2         - 4x2 matrix of vertices for polygon 2

    if nargin < 7 || isempty(margin1), margin1 = 0.0; end
    if nargin < 8 || isempty(margin2), margin2 = 0.0; end

    % 1. Compute 4 corners of OBB 1
    corners1 = compute_obb_corners_internal(center1, size1, yaw1, margin1);
    % 2. Compute 4 corners of OBB 2
    corners2 = compute_obb_corners_internal(center2, size2, yaw2, margin2);

    % 3. SAT Axis Projections
    polygons = {corners1, corners2};
    is_collision = true;
    min_sep = -Inf; % Tracks maximum overlap (minimum separation)

    for p_idx = 1:2
        poly = polygons{p_idx};
        N_vert = size(poly, 1);
        for i = 1:N_vert
            p1 = poly(i, :);
            p2 = poly(mod(i, N_vert) + 1, :);
            edge = p2 - p1;

            % Outward perpendicular normal: [-dy, dx]
            normal = [-edge(2), edge(1)];
            n_len = norm(normal);
            if n_len < 1e-12
                continue;
            end
            normal = normal / n_len;

            % Project both polygons onto axis
            proj1 = corners1 * normal';
            proj2 = corners2 * normal';

            min1 = min(proj1); max1 = max(proj1);
            min2 = min(proj2); max2 = max(proj2);

            % Separation gap: positive if separated, negative if overlapping
            gap = max(min1 - max2, min2 - max1);

            if gap > 0.0
                % Separating axis found!
                is_collision = false;
                min_sep = gap;
                return;
            end

            if gap > min_sep
                min_sep = gap;
            end
        end
    end
end

function corners = compute_obb_corners_internal(center, size_box, yaw, margin)
    hl = (size_box(1) / 2.0) + margin;
    hw = (size_box(2) / 2.0) + margin;

    % Local unrotated corners: [FL, FR, RR, RL]
    local_corners = [ hl,  hw;
                      hl, -hw;
                     -hl, -hw;
                     -hl,  hw];

    c = cos(yaw);
    s = sin(yaw);
    R = [c, -s; s, c];

    corners = (R * local_corners')';
    corners(:, 1) = corners(:, 1) + center(1);
    corners(:, 2) = corners(:, 2) + center(2);
end

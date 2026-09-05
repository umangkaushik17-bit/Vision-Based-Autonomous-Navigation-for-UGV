function p_cam = transform_robot_to_camera(p_robot, R, t)
% TRANSFORM_ROBOT_TO_CAMERA Transforms 3D points from Robot frame to Camera Optical frame.
% Input:
%   p_robot: (3, N) or (N, 3) matrix of coordinates in robot frame (X-fwd, Y-left, Z-up)
%   R: (3, 3) rotation matrix
%   t: (3, 1) translation vector
% Output:
%   p_cam: (3, N) or (N, 3) coordinates in camera optical frame (X-right, Y-down, Z-fwd)

    was_row_format = false;
    if size(p_robot, 1) ~= 3 && size(p_robot, 2) == 3
        p_robot = p_robot';
        was_row_format = true;
    end

    if nargin < 2 || isempty(R) || isempty(t)
        % Load defaults
        calib = validate_extrinsics();
        R = calib.R;
        t = calib.t;
    end

    p_cam = R * p_robot + t;

    if was_row_format
        p_cam = p_cam';
    end
end

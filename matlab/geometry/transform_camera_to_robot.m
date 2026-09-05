function p_robot = transform_camera_to_robot(p_cam, R, t)
% TRANSFORM_CAMERA_TO_ROBOT Transforms 3D points from Camera Optical frame to Robot frame.
% Input:
%   p_cam: (3, N) or (N, 3) coordinates in camera optical frame
%   R: (3, 3) rotation matrix (robot to camera)
%   t: (3, 1) translation vector (robot to camera)
% Output:
%   p_robot: (3, N) or (N, 3) coordinates in robot frame (X-fwd, Y-left, Z-up)

    was_row_format = false;
    if size(p_cam, 1) ~= 3 && size(p_cam, 2) == 3
        p_cam = p_cam';
        was_row_format = true;
    end

    if nargin < 2 || isempty(R) || isempty(t)
        calib = validate_extrinsics();
        R = calib.R;
        t = calib.t;
    end

    p_robot = R' * (p_cam - t);

    if was_row_format
        p_robot = p_robot';
    end
end

# DynamicObstacleDetection.msg
# Single-frame dynamic obstacle detection bounding representation.

std_msgs/Header header
uint32 detection_id
string class_name
float32 confidence

# Bounding box state in header.frame_id
geometry_msgs/Point position
geometry_msgs/Vector3 dimensions  # (length, width, height) in meters
float32 yaw_rad                   # Planar orientation in radians

# Observation uncertainty (2x2 position covariance in row-major order: [xx, xy, yx, yy])
float32[4] position_covariance

bool is_valid

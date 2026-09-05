# TrackedObstacle.msg
# Filtered dynamic obstacle track with persistent ID, velocity, and covariance.

std_msgs/Header header
uint32 track_id
string status          # "NEW", "TRACKING", "DEGRADED", "LOST", "EXPIRED"
bool is_confirmed
float32 confidence

# State: position (x, y, z) and planar velocity (vx, vy, 0)
geometry_msgs/Point position
geometry_msgs/Vector3 velocity

# Dimensions and orientation
geometry_msgs/Vector3 bbox_size   # (length, width, height) in meters
float32 bbox_yaw                  # Planar yaw angle in radians

# 4x4 State Covariance [x, y, vx, vy] in row-major order (16 elements)
float32[16] covariance

# Tracking metrics
uint32 total_hits
uint32 consecutive_misses
float32 time_since_last_update_s

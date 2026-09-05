# PredictedObstacleTrajectory.msg
# Extrapolated forward motion trajectory for a tracked dynamic obstacle.

std_msgs/Header header
uint32 track_id
string status                # "TRACKING", "DEGRADED", "LOST"
geometry_msgs/Vector3 bbox_size
float32 bbox_yaw

TrajectoryWaypoint[] waypoints

float32 prediction_horizon_s
bool is_unknown              # True if prediction extends into unknown regime
float32 confidence

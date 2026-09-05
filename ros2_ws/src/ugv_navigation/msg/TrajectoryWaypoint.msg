# TrajectoryWaypoint.msg
# Single predicted future spatiotemporal waypoint along an obstacle trajectory.

builtin_interfaces/Time target_time
float32 relative_time_s

# Predicted position and velocity
geometry_msgs/Point position
geometry_msgs/Vector3 velocity

# 2x2 Spatial Positional Covariance [xx, xy, yx, yy] (4 elements)
float32[4] covariance

float32 uncertainty_radius_m
float32 confidence

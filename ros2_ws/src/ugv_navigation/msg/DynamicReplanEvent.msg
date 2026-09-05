# DynamicReplanEvent.msg
# Event record emitted during event-driven global replanning (Milestone 13.11).

std_msgs/Header header
string event_id

# Trigger: "DYNAMIC_DEADLOCK", "PERSISTENT_BLOCKAGE", "INSUFFICIENT_PROGRESS", "PATH_INVALIDATED", "MANUAL"
string trigger_type

geometry_msgs/Point current_pose
float32 current_yaw
geometry_msgs/Point current_goal

uint32[] blocking_obstacle_ids
float32 blocked_duration_s
float32 deadlock_duration_s
float32 current_progress_m
string reason
float32 confidence

# Replan outcome
bool success
bool is_path_stale
float32 old_path_length_m
float32 new_path_length_m
float32 path_deviation_m
float32 planning_latency_ms

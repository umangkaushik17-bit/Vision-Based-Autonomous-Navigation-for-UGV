# LocalizationStatus.msg
# Machine-readable localization estimate contract with explicit tracking state and validity.

std_msgs/Header header
string child_frame_id

# Tracking states:
uint8 TRACKING_OK = 0
uint8 DEGRADED = 1
uint8 LOST = 2
uint8 RELOCALIZING = 3
uint8 RECOVERED = 4

uint8 tracking_state
float32 confidence
bool is_pose_valid
bool allows_spatial_fusion

geometry_msgs/PoseWithCovariance pose
geometry_msgs/TwistWithCovariance twist

uint32 tracked_features
uint32 inliers
uint32 keyframe_id
bool is_keyframe
bool relocalized

# WorldTerrainMap.msg
# Canonical 11-layer world-referenced metric terrain map for autonomous UGV navigation.

std_msgs/Header header

float32 resolution
geometry_msgs/Pose origin

uint32 width
uint32 height

float32[] cost
float32[] traversability
bool[] obstacle
float32[] hazard
bool[] unknown
float32[] confidence
float32[] slope
float32[] roughness
float32[] height_var
uint32[] observation_count

# DynamicOccupancyGridSlice.msg
# Spatiotemporal raster occupancy slice for a specific future timestamp.

std_msgs/Header header
float32 target_time_offset_s
builtin_interfaces/Time target_timestamp

float32 resolution
float32 origin_x
float32 origin_y
uint32 width
uint32 height

# Flattened 1D array of cell risk values in row-major order:
# 0.0 = FREE, (0.0, 0.40) = LOW, [0.40, 0.70) = MEDIUM, [0.70, 0.95) = HIGH, >= 0.95 = LETHAL
float32[] data
bool[] unknown_mask

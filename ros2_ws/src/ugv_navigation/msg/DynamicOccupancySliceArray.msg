# DynamicOccupancySliceArray.msg
# Collection of time-sliced dynamic occupancy layers across the prediction horizon.

std_msgs/Header header
DynamicOccupancyGridSlice[] slices

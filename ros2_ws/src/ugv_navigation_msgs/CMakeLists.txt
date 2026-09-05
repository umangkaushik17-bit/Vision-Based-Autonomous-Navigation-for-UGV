cmake_minimum_required(VERSION 3.8)
project(ugv_navigation_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

set(msg_files
  "msg/BehavioralDecision.msg"
  "msg/DynamicObstacleDetection.msg"
  "msg/DynamicObstacleDetectionArray.msg"
  "msg/DynamicOccupancyGridSlice.msg"
  "msg/DynamicOccupancySliceArray.msg"
  "msg/DynamicReplanEvent.msg"
  "msg/LocalizationStatus.msg"
  "msg/PredictedObstacleTrajectory.msg"
  "msg/PredictedTrajectoryArray.msg"
  "msg/TrackedObstacle.msg"
  "msg/TrackedObstacleArray.msg"
  "msg/TrajectoryWaypoint.msg"
  "msg/WorldTerrainMap.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES
    std_msgs
    geometry_msgs
    builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

"""Dynamic obstacle perception, tracking, prediction, and predictive planning modules for Phase 13."""

from ai.planning.dynamic.data_model import (
    DynamicObstacleDetection,
    DynamicObstacleDetectionArray,
    PredictedObstacleTrajectory,
    PredictedTrajectoryArray,
    TrackedObstacleArray,
    TrackedObstacleState,
    TrajectoryWaypoint,
    compute_obb_corners,
    normalize_angle,
    point_in_polygon,
    polygon_intersection_sat,
)
from ai.planning.dynamic.detector_interface import (
    BaseDynamicObstacleDetector,
    SyntheticDynamicObstacleDetector,
    SyntheticObstacleScript,
    VisionDynamicObstacleDetector,
)
from ai.planning.dynamic.association import (
    BaseAssociation,
    HungarianAssociation,
)
from ai.planning.dynamic.tracker import (
    MultiFrameTracker,
    Track,
    TrackStatus,
    TrackerConfig,
)
from ai.planning.dynamic.motion_estimator import (
    MotionEstimate,
    MotionEstimator,
    MotionEstimatorConfig,
)
from ai.planning.dynamic.kalman_filter import (
    KalmanFilterConfig,
    KalmanTrackFilter,
)
from ai.planning.dynamic.predictor import (
    MotionPredictor,
    PredictorConfig,
)
from ai.planning.dynamic.occupancy_layer import (
    DynamicOccupancyConfig,
    DynamicOccupancyGridSlice,
    DynamicOccupancyLayer,
    DynamicRiskLevel,
)
from ai.planning.dynamic.collision_checker import (
    CollisionStatus,
    PredictiveCollisionChecker,
    PredictiveCollisionConfig,
    PredictiveCollisionResult,
    TrajectorySample,
    compute_polygon_clearance,
    point_to_segment_distance,
    sample_ugv_footprint_points,
)
from ai.planning.dynamic.dwa_filter import (
    DynamicBroadPhaseFilter,
    DynamicPlanningConfig,
    ObstacleSpatiotemporalHull,
)

from ai.planning.dynamic.behavioral_state import (
    BehavioralConfig,
    BehavioralDecision,
    BehavioralState,
    BehavioralStateMachine,
    DeadlockEvent,
)
from ai.planning.dynamic.global_replanner import (
    DynamicPlanningMapOverlay,
    DynamicReplanEvent,
    DynamicReplanResult,
    DynamicReplanTrigger,
    EventDrivenGlobalReplanner,
    GlobalReplannerConfig,
    PathChangeMetrics,
    compute_min_dynamic_clearance,
    compute_path_deviation,
)

__all__ = [
    "DynamicObstacleDetection",
    "DynamicObstacleDetectionArray",
    "TrackedObstacleState",
    "TrackedObstacleArray",
    "TrajectoryWaypoint",
    "PredictedObstacleTrajectory",
    "PredictedTrajectoryArray",
    "compute_obb_corners",
    "polygon_intersection_sat",
    "point_in_polygon",
    "normalize_angle",
    "BaseDynamicObstacleDetector",
    "SyntheticDynamicObstacleDetector",
    "SyntheticObstacleScript",
    "VisionDynamicObstacleDetector",
    "BaseAssociation",
    "HungarianAssociation",
    "MultiFrameTracker",
    "Track",
    "TrackStatus",
    "TrackerConfig",
    "MotionEstimate",
    "MotionEstimator",
    "MotionEstimatorConfig",
    "KalmanFilterConfig",
    "KalmanTrackFilter",
    "MotionPredictor",
    "PredictorConfig",
    "DynamicOccupancyLayer",
    "DynamicOccupancyConfig",
    "DynamicOccupancyGridSlice",
    "DynamicRiskLevel",
    "CollisionStatus",
    "PredictiveCollisionChecker",
    "PredictiveCollisionConfig",
    "PredictiveCollisionResult",
    "TrajectorySample",
    "compute_polygon_clearance",
    "point_to_segment_distance",
    "sample_ugv_footprint_points",
    "DynamicPlanningConfig",
    "DynamicBroadPhaseFilter",
    "ObstacleSpatiotemporalHull",
    "BehavioralState",
    "BehavioralConfig",
    "BehavioralDecision",
    "BehavioralStateMachine",
    "DeadlockEvent",
    "DynamicReplanTrigger",
    "DynamicReplanEvent",
    "DynamicReplanResult",
    "PathChangeMetrics",
    "GlobalReplannerConfig",
    "DynamicPlanningMapOverlay",
    "EventDrivenGlobalReplanner",
    "compute_path_deviation",
    "compute_min_dynamic_clearance",
]



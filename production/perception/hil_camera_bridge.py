"""Plug-and-Play Live Webcam / Stereo Hardware-in-the-Loop (HIL) Navigation Bridge (Subphase 16.3).

Enables live physical USB webcam or synthetic test streams at the SIH evaluation table
to drive the full production autonomy stack in real time (>= 28 FPS, < 35 ms E2E):
  RGB Frame -> Photometric LAB-CLAHE -> Traversability tau(u, v) -> SE(3) Attitude IPM
  -> 2.5D WorldTerrainMap -> C++17 A* & C++17 AVX2 DWA -> SafetyGovernor -> 3-Panel HUD
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from ai.perception_fusion.cost_model import CostModelWeights, TraversabilityCostModel
from ai.planning.backends.cpp_backend import CPPDWABackend
from ai.planning.base import GoalPose, PlanningConfig, PlannerStatus, RobotState
from ai.planning.global_planners.astar import AStarPlanner
from ai.planning.path import Path
from ai.planning.safety_governor import SafetyGovernor
from ai.spatial_fusion.world_map import WorldTerrainMap
from production.perception.elevation_mapper import ElevationMapper
from production.perception.photometric_resilience import PhotometricResilienceProcessor
from production.perception.terrain_geometry import VectorizedTerrainGeometryEstimator
from production.perception.vision_costmap_integrator import VisionCostmapIntegrator
from simulation.nodes.presentation_visualization_adapter import JudgeDashboardTelemetryAdapter


@dataclass
class HILStepResult:
    """Per-frame output from HILCameraNavigationBridge."""

    frame_index: int
    timestamp_s: float
    v_cmd: float
    w_cmd: float
    raw_dwa_v: float
    raw_dwa_w: float
    roll_deg: float
    pitch_deg: float
    rollover_margin: float
    safety_decision: str
    behavioral_state: str
    obstacles_marked: int
    obstacle_projected_xy: Optional[Tuple[float, float]]
    total_latency_ms: float
    fps: float
    composite_hud: Optional[np.ndarray]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": round(self.timestamp_s, 3),
            "v_cmd": round(self.v_cmd, 3),
            "w_cmd": round(self.w_cmd, 3),
            "roll_deg": round(self.roll_deg, 2),
            "pitch_deg": round(self.pitch_deg, 2),
            "rollover_margin": round(self.rollover_margin, 3),
            "safety_decision": self.safety_decision,
            "behavioral_state": self.behavioral_state,
            "obstacles_marked": self.obstacles_marked,
            "obstacle_projected_xy": (
                [round(self.obstacle_projected_xy[0], 3), round(self.obstacle_projected_xy[1], 3)]
                if self.obstacle_projected_xy is not None
                else None
            ),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "fps": round(self.fps, 1),
        }


class SyntheticCameraSource:
    """Deterministic synthetic outdoor camera stream with obstacle & SE(3) tilt injection for CI/HIL testing."""

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        integrator: Optional[VisionCostmapIntegrator] = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.integrator = integrator or VisionCostmapIntegrator.create_simulation_default()
        self.frame_idx = 0

        yy, xx = np.mgrid[0:self.height, 0:self.width]
        # Natural outdoor sky-to-trail vertical gradient + texture with nominal contrast (sigma_L ~ 32, mean_L ~ 125)
        grad = ((yy.astype(np.float32) / float(self.height)) - 0.5) * 90.0
        wave = 18.0 * np.sin(xx * 0.09) + 14.0 * np.cos(yy * 0.07)
        r = np.clip(118.0 + grad + wave, 62.0, 210.0).astype(np.uint8)
        g = np.clip(138.0 + grad + wave, 72.0, 225.0).astype(np.uint8)
        b = np.clip(96.0 + grad + wave, 52.0, 185.0).astype(np.uint8)
        self._base_rgb = np.stack([r, g, b], axis=-1)

    def read_frame(
        self,
        obstacle_ground_xy: Optional[Tuple[float, float]] = None,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
        glare_boost: float = 0.0,
    ) -> np.ndarray:
        """Render synthetic outdoor trail RGB frame with optional ground obstacle and chassis attitude."""
        self.frame_idx += 1
        frame = self._base_rgb.copy()
        if abs(glare_boost) > 1e-3:
            frame = np.clip(frame.astype(np.float32) * (1.0 + glare_boost), 0, 255).astype(np.uint8)

        if obstacle_ground_xy is not None:
            ox, oy = obstacle_ground_xy
            px = self.integrator.project_ground_point_to_pixel(
                ox,
                oy,
                roll_rad=math.radians(roll_deg),
                pitch_rad=math.radians(pitch_deg),
                image_shape=(self.height, self.width),
            )
            if px is not None:
                vo, uo = px
                r0 = max(0, vo - 1)
                r1 = min(self.height, vo + 2)
                c0 = max(0, uo - 2)
                c1 = min(self.width, uo + 3)
                # Distinct red physical obstacle marker
                frame[r0:r1, c0:c1] = np.array([235, 15, 15], dtype=np.uint8)

        return frame


class OpenCVCameraSource:
    """Live USB webcam source with automatic fallback to SyntheticCameraSource when headless."""

    def __init__(self, camera_id: int = 0, width: int = 320, height: int = 240, allow_fallback: bool = True) -> None:
        self.camera_id = int(camera_id)
        self.width = int(width)
        self.height = int(height)
        self.cap: Optional[Any] = None
        self.synthetic_fallback = SyntheticCameraSource(width=self.width, height=self.height)
        self.using_fallback = True

        try:
            cap = cv2.VideoCapture(self.camera_id)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap = cap
                self.using_fallback = False
            elif not allow_fallback:
                raise RuntimeError(f"Could not open USB camera index {self.camera_id}")
        except Exception:
            if not allow_fallback:
                raise
            self.cap = None
            self.using_fallback = True

    def read(self, **synthetic_kwargs: Any) -> np.ndarray:
        if self.cap is not None and not self.using_fallback:
            ok, bgr = self.cap.read()
            if ok and bgr is not None and bgr.size > 0:
                resized = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
                return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return self.synthetic_fallback.read_frame(**synthetic_kwargs)

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass


class HILCameraNavigationBridge:
    """End-to-end Hardware-in-the-Loop camera navigation bridge and 3-panel diagnostic renderer."""

    def __init__(self) -> None:
        self.photometric = PhotometricResilienceProcessor()
        self.integrator = VisionCostmapIntegrator.create_simulation_default()
        self.elevation_mapper = ElevationMapper()
        self.terrain_estimator = VectorizedTerrainGeometryEstimator()
        self.cost_model = TraversabilityCostModel(CostModelWeights.from_yaml("configs/terrain_safety.yaml"))
        self.safety_governor = SafetyGovernor()

        self.cfg = PlanningConfig()
        self.cfg.vehicle.max_linear_velocity_mps = 0.70
        self.cfg.vehicle.min_linear_velocity_mps = 0.0
        self.cfg.dwa["prediction_horizon_s"] = 1.50
        self.cfg.dwa["simulation_dt_s"] = 0.10

        self.astar = AStarPlanner(config=self.cfg, backend="cpp")
        self.dwa = CPPDWABackend(config=self.cfg, variant="auto")

        self.world_map = WorldTerrainMap(
            resolution_m=0.10,
            x_bounds_m=(-1.0, 7.0),
            y_bounds_m=(-3.5, 3.5),
        )
        self.reset_map()
        self.frame_counter = 0

    def reset_map(self) -> None:
        self.world_map.unknown.fill(False)
        self.world_map.cost.fill(0.08)
        self.world_map.traversability.fill(0.92)
        self.world_map.obstacle.fill(False)
        self.world_map.hazard.fill(0.0)
        self.world_map.elevation.fill(0.0)
        self.world_map.elevation_valid.fill(True)
        self.world_map.slope.fill(0.0)
        self.world_map.height_var.fill(0.0)
        self.world_map._cached_infl_obstacle = None
        self.integrator._obstacle_clear_counter.clear()

    def step_frame(
        self,
        rgb: np.ndarray,
        timestamp_s: Optional[float] = None,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
        goal_xy: Tuple[float, float] = (5.5, 0.0),
        reset_world_before_step: bool = True,
        render_hud: bool = True,
    ) -> HILStepResult:
        """Execute one complete closed-loop HIL perception-to-control step."""
        t0 = time.perf_counter()
        self.frame_counter += 1
        ts = timestamp_s if timestamp_s is not None else time.time()

        if reset_world_before_step:
            self.reset_map()

        roll_rad = math.radians(roll_deg)
        pitch_rad = math.radians(pitch_deg)

        # 1. Photometric LAB-CLAHE + Traversability Prediction
        trav_res, _ = self.photometric.evaluate_outdoor_traversability(rgb, timestamp_s=ts)

        # 2. SE(3) Attitude-Compensated BEV Projection & WorldMap Fusion
        bev = self.integrator.project_traversability(
            trav_res,
            current_time_s=ts,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
        )
        fusion_stats = self.integrator.fuse_into_world_map(
            bev,
            robot_pose=(0.0, 0.0, 0.0),
            world_map=self.world_map,
        )
        self.world_map._cached_infl_obstacle = None

        obs_r, obs_c = np.where(bev.obstacle)
        obs_proj_xy: Optional[Tuple[float, float]] = None
        if len(obs_r) > 0:
            obs_proj_xy = (
                float(np.mean(bev.x_robot[obs_r, obs_c])),
                float(np.mean(bev.y_robot[obs_r, obs_c])),
            )

        # 3. C++17 Global A* Path Planning
        start_state = RobotState(x=0.0, y=0.0, yaw=0.0, v=0.35, omega=0.0, timestamp=ts)
        goal_pose = GoalPose(x=goal_xy[0], y=goal_xy[1])
        global_path, astar_status, _ = self.astar.plan(self.world_map, start_state, goal_pose)
        if astar_status != PlannerStatus.SUCCESS or global_path is None:
            global_path = Path.from_xy_array(np.array([[0.0, 0.0], [goal_xy[0], goal_xy[1]]], dtype=np.float32))

        # 4. C++17 AVX2 Local DWA Rollout Evaluation
        best_traj, raw_v, raw_w, _, dwa_diag = self.dwa.compute_trajectory(
            current_state=start_state,
            global_path=global_path,
            costmap=self.world_map,
        )
        beh_state = str(dwa_diag.get("behavioral_state", "PROCEED"))

        # 5. Authoritative 3D Terrain & Attitude Safety Governor
        gov_v, gov_w, gov_dec, _ = self.safety_governor.evaluate(
            requested_v=raw_v,
            requested_w=raw_w,
            robot_pose=(0.0, 0.0, 0.0),
            hazards=[],
            current_time_s=ts,
            robot_roll_pitch=(roll_rad, pitch_rad),
            costmap=self.world_map,
        )

        m_roll = JudgeDashboardTelemetryAdapter.compute_rollover_margin(
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            linear_v_mps=gov_v,
            angular_w_radps=gov_w,
        )

        hud_img: Optional[np.ndarray] = None
        if render_hud:
            hud_img = self._render_composite_hud(
                rgb=rgb,
                trav_mask=trav_res.mask,
                global_path=global_path,
                best_traj=best_traj,
                v_cmd=gov_v,
                w_cmd=gov_w,
                roll_deg=roll_deg,
                pitch_deg=pitch_deg,
                rollover_margin=m_roll,
                safety_decision=gov_dec,
                fps=1000.0 / max(0.5, (time.perf_counter() - t0) * 1000.0),
            )

        total_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / max(1e-3, total_ms)

        return HILStepResult(
            frame_index=self.frame_counter,
            timestamp_s=ts,
            v_cmd=gov_v,
            w_cmd=gov_w,
            raw_dwa_v=raw_v,
            raw_dwa_w=raw_w,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            rollover_margin=m_roll,
            safety_decision=gov_dec,
            behavioral_state=beh_state,
            obstacles_marked=int(fusion_stats.get("obstacles_marked", 0)),
            obstacle_projected_xy=obs_proj_xy,
            total_latency_ms=total_ms,
            fps=fps,
            composite_hud=hud_img,
        )

    def _render_composite_hud(
        self,
        rgb: np.ndarray,
        trav_mask: np.ndarray,
        global_path: Path,
        best_traj: Any,
        v_cmd: float,
        w_cmd: float,
        roll_deg: float,
        pitch_deg: float,
        rollover_margin: float,
        safety_decision: str,
        fps: float,
    ) -> np.ndarray:
        """Render 3-panel split-screen OpenCV diagnostic composite (280 x 640 x 3 uint8)."""
        canvas = np.zeros((280, 640, 3), dtype=np.uint8)

        # Panel 1 (Left 240x320): Live RGB + Traversability Overlay + Projected Best DWA Arc
        p1 = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_NEAREST).copy()
        obs_px = trav_mask < 0.30
        if np.any(obs_px):
            p1[obs_px] = np.array([230, 40, 40], dtype=np.uint8)
        if best_traj is not None and hasattr(best_traj, "points"):
            pts_uv = []
            for tp in best_traj.points[::2]:
                uv = self.integrator.project_ground_point_to_pixel(tp.x, tp.y, image_shape=(240, 320))
                if uv is not None:
                    pts_uv.append((uv[1], uv[0]))
            for i in range(len(pts_uv) - 1):
                cv2.line(p1, pts_uv[i], pts_uv[i + 1], (0, 255, 180), 2)
        canvas[0:240, 0:320] = p1

        # Panel 2 (Right 240x320): 2.5D BEV Costmap + Obstacle Hazards + Global A* Path
        cost_u8 = (np.clip(self.world_map.cost, 0.0, 1.0) * 255.0).astype(np.uint8)
        p2_gray = cv2.resize(cost_u8, (320, 240), interpolation=cv2.INTER_NEAREST)
        p2 = cv2.cvtColor(p2_gray, cv2.COLOR_GRAY2RGB)
        if len(global_path.points) >= 2:
            xy = global_path.waypoints_xy
            rows, cols = self.world_map.world_to_grid_vec(xy[:, 0], xy[:, 1])
            us = np.clip((cols * (320.0 / self.world_map.n_cols)).astype(np.int32), 0, 319)
            vs = np.clip((rows * (240.0 / self.world_map.n_rows)).astype(np.int32), 0, 239)
            for i in range(len(us) - 1):
                cv2.line(p2, (int(us[i]), int(vs[i])), (int(us[i + 1]), int(vs[i + 1])), (40, 240, 80), 2)
        canvas[0:240, 320:640] = p2

        # Panel 3 (Bottom 40x640): Live Telemetry Strip
        canvas[240:280, :] = (24, 28, 36)
        strip_txt = (
            f"FPS:{fps:4.1f} | v:{v_cmd:.2f}m/s w:{w_cmd:+.2f}rad/s | "
            f"Roll:{roll_deg:+.1f} Pitch:{pitch_deg:+.1f} | M_roll:{rollover_margin:.2f} | {safety_decision[:22]}"
        )
        cv2.putText(canvas, strip_txt, (8, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 245, 255), 1)
        return canvas

"""
NWM Navigation Node  —  runs on the workstation.

Subscribes to the LIMO's ROS2 topics directly (same DDS domain),
runs NWM planning on the GPU, and publishes /cmd_vel back.

Prerequisites on workstation:
    pip install pyzmq lpips  (pyzmq not needed but lpips is)
    # ROS2 must be sourced, e.g.:  source /opt/ros/humble/setup.bash
    # Robot and workstation must share the same ROS_DOMAIN_ID

Usage:
    cd /home/suyoung/Documents/Git/nwm/online_nav

    # Option A: provide a saved goal image
    python online_nav/nwm_nav_node.py \
        --ckp ./logs/nwm_cdit_xl/checkpoints/cdit_xl_ego4d_200000.pth.tar \
        --goal_image ./online_nav/goal_test_1.jpg

    # Option B: capture goal live (drive robot to goal position, press Enter)
    python online_nav/nwm_nav_node.py \
        --ckp ./logs/nwm_cdit_xl/checkpoints/cdit_xl_ego4d_200000.pth.tar \
        --capture_goal

    # Full options:
    python online_nav/nwm_nav_node.py \
        --ckp ./logs/nwm_cdit_xl/checkpoints/cdit_xl_ego4d_200000.pth.tar \
        --goal_image goal.jpg \
        --image_topic /camera/color/image_raw \
        --cmd_vel_topic /limo2/cmd_vel \
        --num_samples 30 --diffusion_steps 50 \
        --rollout_stride 4 --len_traj_pred 8 \
        --metric_waypoint_spacing 0.05 \
        --plan_hz 0.3 --exec_duration 1.0 \
        --max_linear 0.2 --max_angular 0.8
"""

import argparse
import threading
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import Twist

from nwm_planner import NWMPlanner


# ---------------------------------------------------------------------------
# Differential-drive controller
# ---------------------------------------------------------------------------

class DiffDriveController:
    """
    P controller for differential drive.
    Given a desired (dx, dy) in the robot frame, returns (linear_vel, angular_vel).
    """
    def __init__(self, max_linear=0.2, max_angular=0.8, k_heading=1.5, k_dist=0.5):
        self.max_linear  = max_linear
        self.max_angular = max_angular
        self.k_heading   = k_heading
        self.k_dist      = k_dist

    def compute(self, dx: float, dy: float, model_fps: float = 4.0) -> tuple[float, float]:
        dist          = float(np.hypot(dx, dy))
        heading_error = float(np.arctan2(dy, dx))   # angle to waypoint, robot frame

        # dx/dy are per-frame displacements (meters/frame at model_fps); convert to m/s
        linear  = min(self.max_linear, dist * model_fps)
        angular = float(np.clip(self.k_heading * heading_error,
                                -self.max_angular, self.max_angular))
        # Rotate in place if heading error is large before moving forward
        if abs(heading_error) > np.radians(30):
            linear = 0.0
        return linear, angular


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class NWMNavNode(Node):

    def __init__(self, args):
        super().__init__('nwm_nav')
        self.args   = args
        self.controller = DiffDriveController(
            max_linear=args.max_linear,
            max_angular=args.max_angular,
        )

        # Latest camera frame (thread-safe)
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()

        # Visualization state (thread-safe, updated from callbacks / planner)
        self._viz_lock     = threading.Lock()
        self._viz_goal     : np.ndarray | None = None   # RGB, set once
        self._viz_obs      : np.ndarray | None = None   # RGB, updated per frame
        self._viz_imagined : np.ndarray | None = None   # RGB, updated per plan step
        self._viz_elapsed  : float = 0.0
        self._viz_dx       : float = 0.0
        self._viz_dy       : float = 0.0

        # ROS2 pub / sub
        self._img_sub  = self.create_subscription(
            RosImage, args.image_topic, self._image_cb, 10)
        self._cmd_pub  = self.create_publisher(Twist, args.cmd_vel_topic, 10)

        # Load NWM on the GPU (this blocks for ~10-20s on first torch.compile)
        self.get_logger().info('Loading NWM model...')
        self.planner = NWMPlanner(
            ckp_path=args.ckp,
            num_samples=args.num_samples,
            topk=args.topk,
            opt_steps=args.opt_steps,
            rollout_stride=args.rollout_stride,
            len_traj_pred=args.len_traj_pred,
            diffusion_steps=args.diffusion_steps,
            num_repeat_eval=args.num_repeat_eval,
            mu_init=[args.mu_x, 0.0, 0.0],
            sigma_init=[args.sigma_x, 0.1, 0.1],
            metric_waypoint_spacing=args.metric_waypoint_spacing,
        )

        # Register goal (blocks until user is ready if --capture_goal)
        self._register_goal()

        # Start MPC timer
        self._plan_timer = self.create_timer(1.0 / args.plan_hz, self._plan_and_execute)
        self.get_logger().info('Navigation started.')

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _image_cb(self, msg: RosImage):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == 'bgr8':
            frame = frame[:, :, ::-1].copy()
        elif msg.encoding not in ('rgb8', 'rgba8'):
            self.get_logger().warn(f'Unexpected image encoding: {msg.encoding}', throttle_duration_sec=5)
        if frame.shape[2] == 4:   # drop alpha channel if present
            frame = frame[:, :, :3]
        with self._frame_lock:
            self._latest_frame = frame
        with self._viz_lock:
            self._viz_obs = frame

    def _get_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def _wait_for_frame(self, timeout: float = 10.0) -> np.ndarray:
        """Block until at least one camera frame is available."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            frame = self._get_frame()
            if frame is not None:
                return frame
        raise TimeoutError(f'No frame received on {self.args.image_topic} after {timeout}s')

    # ------------------------------------------------------------------
    # Goal registration
    # ------------------------------------------------------------------

    def _register_goal(self):
        if self.args.goal_image:
            bgr = cv2.imread(self.args.goal_image)
            if bgr is None:
                raise FileNotFoundError(f'Cannot read goal image: {self.args.goal_image}')
            goal_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.get_logger().info(f'Goal image loaded from {self.args.goal_image}')
        else:
            # Capture live: drive robot to goal, press Enter
            print('\n[NWM] Drive the robot to the GOAL position, then press Enter...')
            input()
            goal_img = self._wait_for_frame()
            print('[NWM] Goal captured. Drive robot back to start, then press Enter...')
            input()

        self.planner.set_goal(goal_img)
        with self._viz_lock:
            self._viz_goal = goal_img
        # Pre-fill context buffer with whatever the robot sees right now
        start_frame = self._wait_for_frame()
        for _ in range(self.planner.context_size):
            self.planner.push_frame(start_frame)
        self.get_logger().info('Goal registered, context buffer initialised.')

    # ------------------------------------------------------------------
    # MPC loop  (called by timer at plan_hz)
    # ------------------------------------------------------------------

    def _plan_and_execute(self):
        frame = self._get_frame()
        if frame is None:
            self.get_logger().warn('No camera frame yet, skipping planning step.')
            return

        # 1. Push new observation into planner context
        self.planner.push_frame(frame)

        # 2. Run NWM planning (blocking; takes a few seconds without distillation)
        t0 = time.time()
        try:
            step_actions, total_dyaw, debug_img = self.planner.plan()
        except Exception as e:
            self.get_logger().error(f'Planning failed: {e}')
            return
        elapsed = time.time() - t0

        # step_actions: (len_traj_pred, 2) meters, all rows identical (straight-line)
        dx = float(step_actions[0, 0])   # forward  (m)
        dy = float(step_actions[0, 1])   # lateral  (m, positive = left)
        self.get_logger().info(
            f'Plan ({elapsed:.1f}s): dx={dx:.3f}m  dy={dy:.3f}m  dyaw={total_dyaw:.3f}rad')

        # Update visualization state
        with self._viz_lock:
            self._viz_imagined = debug_img
            self._viz_elapsed  = elapsed
            self._viz_dx       = dx
            self._viz_dy       = dy

        # Optionally save the debug image
        if self.args.save_debug:
            cv2.imwrite('nwm_debug.jpg', cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

        # 3. Execute planned velocity for exec_duration seconds
        linear, angular = self.controller.compute(dx, dy)
        self._drive(linear, angular, self.args.exec_duration)

    def _drive(self, linear: float, angular: float, duration: float):
        twist = Twist()
        twist.linear.x  = linear
        twist.angular.z = angular
        end_time = time.time() + duration
        while time.time() < end_time:
            self._cmd_pub.publish(twist)
            time.sleep(0.05)
        self._cmd_pub.publish(Twist())   # stop

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def build_display(self) -> np.ndarray:
        """
        Build a 3-panel BGR image for cv2.imshow:
          [ GOAL | OBSERVATION | IMAGINED ]
        Each panel is 300×300 with a 30-px label bar on top.
        """
        W, H, BAR = 300, 300, 30

        def _placeholder(text: str) -> np.ndarray:
            img = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.putText(img, text, (10, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
            return img

        def _to_panel(rgb: np.ndarray | None, label: str, sub: str = '') -> np.ndarray:
            if rgb is None:
                body = _placeholder('waiting...')
            else:
                body = cv2.cvtColor(
                    cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR),
                    cv2.COLOR_RGB2BGR,
                )
            # Label bar
            bar = np.zeros((BAR, W, 3), dtype=np.uint8)
            cv2.putText(bar, label, (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if sub:
                cv2.putText(bar, sub, (W // 2, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1)
            return np.vstack([bar, body])

        with self._viz_lock:
            goal      = self._viz_goal
            obs       = self._viz_obs
            imagined  = self._viz_imagined
            elapsed   = self._viz_elapsed
            dx        = self._viz_dx
            dy        = self._viz_dy

        plan_sub = f'{elapsed:.1f}s  dx={dx:.2f} dy={dy:.2f}' if imagined is not None else ''

        panel = np.hstack([
            _to_panel(goal,     'GOAL'),
            _to_panel(obs,      'OBSERVATION'),
            _to_panel(imagined, 'NWM IMAGINED', plan_sub),
        ])
        return panel

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def stop(self):
        self._cmd_pub.publish(Twist())
        self.get_logger().info('Robot stopped.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # Model
    p.add_argument('--ckp',         required=True, help='path to .pth.tar checkpoint')
    p.add_argument('--device',      default='cuda')
    # ROS topics
    p.add_argument('--image_topic',    default='/camera/color/image_raw')
    p.add_argument('--cmd_vel_topic',  default='/limo_2/cmd_vel')
    # Goal
    p.add_argument('--goal_image',  default=None, help='path to goal image (.jpg/.png)')
    p.add_argument('--capture_goal', action='store_true',
                   help='capture goal image live (robot driven to goal position first)')
    # Planner hyperparams
    p.add_argument('--num_samples',     type=int,   default=30)
    p.add_argument('--topk',            type=int,   default=5)
    p.add_argument('--opt_steps',       type=int,   default=1)
    p.add_argument('--rollout_stride',  type=int,   default=2)
    p.add_argument('--len_traj_pred',   type=int,   default=2)
    p.add_argument('--diffusion_steps', type=int,   default=50)
    p.add_argument('--num_repeat_eval', type=int,   default=1)
    p.add_argument('--mu_x',            type=float, default=-0.1)
    p.add_argument('--sigma_x',         type=float, default=0.05)
    p.add_argument('--metric_waypoint_spacing', type=float, default=0.05)
    # Controller
    p.add_argument('--max_linear',   type=float, default=0.5,  help='m/s')
    p.add_argument('--max_angular',  type=float, default=0.8,  help='rad/s')
    # Timing
    p.add_argument('--plan_hz',       type=float, default=0.3,
                   help='Planning frequency. Must be achievable by GPU. '
                        'With 30 samples / 50 steps, expect ~3-5s/plan → 0.2-0.3 Hz')
    p.add_argument('--exec_duration', type=float, default=0.5,
                   help='Seconds to execute each planned step before replanning')
    # Debug
    p.add_argument('--save_debug', action='store_true',
                   help='Save NWM predicted final frame to nwm_debug.jpg each step')
    return p.parse_args()


def main():
    rclpy.init()
    args = parse_args()

    if args.goal_image is None and not args.capture_goal:
        raise SystemExit('Provide --goal_image or --capture_goal')

    node = NWMNavNode(args)

    # Spin ROS in a background thread so the main thread is free for cv2.imshow
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    cv2.namedWindow('NWM Navigation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('NWM Navigation', 900, 330)

    try:
        while rclpy.ok() and spin_thread.is_alive():
            panel = node.build_display()
            cv2.imshow('NWM Navigation', panel)
            key = cv2.waitKey(50)   # ~20 Hz refresh
            if key == ord('q') or key == 27:   # q or Esc to quit
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

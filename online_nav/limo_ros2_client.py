"""
LIMO ROS2 Navigation Client  (runs on the robot or a connected machine with ROS2)

What it does:
  1. Subscribes to /camera/image_raw for observations.
  2. On startup, waits for the user to place the robot at the GOAL position,
     captures one frame as the goal image, then moves the robot back to start.
  3. In a timed loop, sends the latest image to the NWM server, receives
     the planned (dx, dy, dyaw), and executes it as /cmd_vel.

MPC execution strategy:
  - Each planning cycle: capture frame → send to server → receive (dx, dy, dyaw)
  - Convert dx, dy (meters, robot frame) to linear + angular velocity
  - Publish /cmd_vel for `exec_duration` seconds, then replan

Install dependencies on the LIMO side:
    pip install pyzmq opencv-python

Usage:
    python limo_ros2_client.py \
        --server_ip 192.168.1.100 --server_port 5555 \
        --goal_image /path/to/goal.jpg \   # OR use --capture_goal flag
        [--max_linear 0.25] [--max_angular 1.0] \
        [--exec_duration 1.0] [--plan_hz 1.0]
"""

import argparse
import pickle
import time
import threading

import cv2
import numpy as np
import zmq

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


# ---------------------------------------------------------------------------
# Simple differential-drive controller
# ---------------------------------------------------------------------------

class DiffDriveController:
    """
    Pure-pursuit-style P controller for differential drive.
    Given a target (dx, dy) in the robot frame, outputs (linear, angular)
    velocities to drive there.
    """
    def __init__(self, max_linear=0.25, max_angular=1.0, k_heading=1.5, k_dist=0.5):
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.k_heading = k_heading
        self.k_dist = k_dist

    def compute(self, dx: float, dy: float) -> tuple[float, float]:
        dist = float(np.hypot(dx, dy))
        # angle to target in robot frame (positive = turn left)
        heading_error = float(np.arctan2(dy, dx))

        linear  = min(self.max_linear, self.k_dist * dist)
        angular = float(np.clip(self.k_heading * heading_error,
                                -self.max_angular, self.max_angular))
        # If heading error is large, rotate first before moving forward
        if abs(heading_error) > np.radians(30):
            linear = 0.0
        return linear, angular


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class LimoNavNode(Node):

    def __init__(self, args):
        super().__init__('limo_nwm_nav')
        self.args = args
        self.bridge = CvBridge()
        self.controller = DiffDriveController(
            max_linear=args.max_linear,
            max_angular=args.max_angular,
        )

        # Latest image from camera (protected by lock)
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()

        # ZMQ client
        ctx = zmq.Context()
        self._sock = ctx.socket(zmq.REQ)
        self._sock.connect(f'tcp://{args.server_ip}:{args.server_port}')
        self.get_logger().info(f'Connected to NWM server at {args.server_ip}:{args.server_port}')

        # ROS2 pub/sub
        self._img_sub = self.create_subscription(
            RosImage, args.image_topic, self._image_callback, 10)
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer that fires the plan-and-execute loop
        period = 1.0 / args.plan_hz
        self._plan_timer = self.create_timer(period, self._plan_and_execute)

        # Send goal image to server
        goal_img = self._load_goal(args)
        self._send_goal(goal_img)
        self.get_logger().info('Goal registered. Navigation started.')

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _image_callback(self, msg: RosImage):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        with self._frame_lock:
            self._latest_frame = frame

    def _get_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    # ------------------------------------------------------------------
    # Goal setup
    # ------------------------------------------------------------------

    def _load_goal(self, args) -> np.ndarray:
        if args.goal_image:
            img = cv2.imread(args.goal_image)
            if img is None:
                raise FileNotFoundError(f'Cannot read goal image: {args.goal_image}')
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Capture goal live: wait for user to position robot, then press Enter
        self.get_logger().info('Move the robot to the GOAL position, then press Enter...')
        input()
        frame = None
        while frame is None:
            frame = self._get_frame()
            time.sleep(0.05)
        self.get_logger().info('Goal captured. Move robot back to start, then press Enter...')
        input()
        return frame

    def _send_goal(self, goal_img: np.ndarray):
        frame = self._get_frame()
        if frame is None:
            self.get_logger().warn('No camera frame available for goal registration, using goal image as context.')
            frame = goal_img
        msg = pickle.dumps({'type': 'set_goal', 'image': frame, 'goal_image': goal_img})
        self._sock.send(msg)
        reply = pickle.loads(self._sock.recv())
        if reply['status'] != 'ok':
            raise RuntimeError(f'Server error on set_goal: {reply.get("message")}')

    # ------------------------------------------------------------------
    # MPC loop
    # ------------------------------------------------------------------

    def _plan_and_execute(self):
        frame = self._get_frame()
        if frame is None:
            self.get_logger().warn('No camera frame yet, skipping.')
            return

        # 1. Send current observation to server
        msg = pickle.dumps({'type': 'step', 'image': frame})
        self._sock.send(msg)
        reply = pickle.loads(self._sock.recv())

        if reply['status'] != 'ok':
            self.get_logger().error(f"Server error: {reply.get('message')}")
            return

        dx = reply['step_dx']    # meters in robot frame (forward)
        dy = reply['step_dy']    # meters in robot frame (left)
        self.get_logger().info(f'Plan: dx={dx:.3f}m  dy={dy:.3f}m  dyaw={reply["total_dyaw"]:.3f}rad')

        # 2. Execute for exec_duration seconds
        linear, angular = self.controller.compute(dx, dy)
        end_time = time.time() + self.args.exec_duration

        twist = Twist()
        twist.linear.x  = linear
        twist.angular.z = angular
        while time.time() < end_time:
            self._cmd_pub.publish(twist)
            time.sleep(0.05)

        # 3. Stop
        self._cmd_pub.publish(Twist())

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def stop(self):
        self._cmd_pub.publish(Twist())
        self.get_logger().info('Robot stopped.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--server_ip',   required=True)
    p.add_argument('--server_port', type=int, default=5555)
    p.add_argument('--image_topic', default='/camera/color/image_raw')
    p.add_argument('--goal_image',  default=None,
                   help='Path to goal image file. If omitted, capture live.')
    # Controller params
    p.add_argument('--max_linear',   type=float, default=0.25, help='m/s')
    p.add_argument('--max_angular',  type=float, default=1.0,  help='rad/s')
    # Timing
    p.add_argument('--plan_hz',       type=float, default=0.5,
                   help='How often to call the NWM server (Hz). '
                        'Must be low enough for the server to finish planning. '
                        'With 30 samples / 50 diffusion steps, expect ~3-5s/plan.')
    p.add_argument('--exec_duration', type=float, default=1.0,
                   help='Seconds to execute each planned step before replanning.')
    return p.parse_args()


def main():
    rclpy.init()
    args = parse_args()
    node = LimoNavNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

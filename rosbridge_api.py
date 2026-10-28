"""Robot camera, laser scan, and velocity interface implemented with roslibpy."""

from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
import roslibpy


@dataclass(frozen=True)
class LaserScan:
    """One decoded ``sensor_msgs/msg/LaserScan`` sample.

    Angles are in radians and ranges are in metres. Invalid range readings are
    retained as ``nan`` or infinity so callers can filter them as appropriate.
    """

    angles: np.ndarray
    ranges: np.ndarray
    intensities: np.ndarray
    angle_min: float
    angle_max: float
    angle_increment: float
    time_increment: float
    scan_time: float
    range_min: float
    range_max: float


class RosbridgeRobot:
    """Access a robot's velocity command, camera, and laser scan topics.

    Args:
        host: Hostname or IP address of the robot's rosbridge server.
        port: rosbridge WebSocket port.
        cmd_vel_topic: Topic accepting ``geometry_msgs/msg/Twist``.
        image_topic: Topic publishing ``sensor_msgs/msg/CompressedImage``.
        connection_timeout: Maximum seconds to wait while connecting.
        laser_scan_topic: Topic publishing ``sensor_msgs/msg/LaserScan``.
        joint_state_topic: Topic publishing ``sensor_msgs/msg/JointState``.
    """

    def __init__(
        self,
        host: str,
        port: int = 9090,
        cmd_vel_topic: str = "/cmd_vel",
        image_topic: str = "/camera/image_raw/compressed",
        connection_timeout: float = 10.0,
        laser_scan_topic: str = "/scan",
        joint_state_topic: str = "/joint_states",
    ) -> None:
        self.host = host
        self.port = port
        self.cmd_vel_topic = cmd_vel_topic
        self.image_topic = image_topic
        self.connection_timeout = connection_timeout
        self.laser_scan_topic = laser_scan_topic
        self.joint_state_topic = joint_state_topic

        self._client: Optional[roslibpy.Ros] = None
        self._publishers: dict[str, roslibpy.Topic] = {}
        self._subscribers: dict[str, roslibpy.Topic] = {}
        self._joint_states: Optional[dict] = None
        self._latest_image: Optional[np.ndarray] = None
        self._image_condition = threading.Condition()
        self._image_sequence = 0
        self._last_query_sequence = 0
        self._latest_laser_scan: Optional[LaserScan] = None
        self._laser_scan_condition = threading.Condition()
        self._laser_scan_sequence = 0
        self._last_laser_scan_query_sequence = 0
        self._is_setup = False

    @property
    def is_connected(self) -> bool:
        """Return whether setup completed and rosbridge remains connected."""
        return bool(
            self._is_setup
            and self._client is not None
            and self._client.is_connected
        )

    def setup(self) -> None:
        """Connect and prepare the command, camera, joint, and scan topics."""
        if self._is_setup:
            return

        self._client = roslibpy.Ros(host=self.host, port=self.port)
        try:
            self._client.run(timeout=self.connection_timeout)
            if not self._client.is_connected:
                raise ConnectionError(
                    f"Could not connect to rosbridge at {self.host}:{self.port}"
                )

            self._add_publisher(
                "cmd_vel", self.cmd_vel_topic, "geometry_msgs/msg/Twist"
            )

            self._add_subscriber(
                "image",
                self.image_topic,
                "sensor_msgs/msg/CompressedImage",
                self._on_image,
            )

            self._add_subscriber(
                "laser_scan",
                self.laser_scan_topic,
                "sensor_msgs/msg/LaserScan",
                self._on_laser_scan,
            )

            self._add_subscriber(
                "joint_states",
                self.joint_state_topic,
                "sensor_msgs/msg/JointState",
                self._on_joint_state,
            )

            self._is_setup = True
        except Exception:
            self.cleanup()
            raise

    def send_command(self, linear: float = 0.0, angular: float = 0.0) -> None:
        """Send linear-x (m/s) and angular-z (rad/s) velocity commands."""
        self._require_connection()
        self._publish_command(linear, angular)

    def stop(self) -> None:
        """Send an immediate zero-velocity command."""
        self.send_command(0.0, 0.0)

    def query_joint_states(self) -> Optional[dict[str, float]]:
        """Return the latest joint states received from the robot.

        Returns:
            A dictionary containing the joint states, or ``None`` if no joint
            states have been received yet.
        """
        self._require_connection()
        return self._joint_states

    def query_laser_scan(
        self, timeout: Optional[float] = None, *, wait_for_new: bool = False
    ) -> Optional[LaserScan]:
        """Return the latest laser scan with NumPy arrays copied for the caller.

        Args:
            timeout: Seconds to wait for a suitable scan. ``None`` waits
                indefinitely and ``0`` returns immediately.
            wait_for_new: Wait for a scan received after the preceding query.

        Returns:
            A :class:`LaserScan`, or ``None`` if the timeout expires or cleanup
            interrupts the query.
        """
        self._require_connection()
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._laser_scan_condition:
            while self._latest_laser_scan is None or (
                wait_for_new
                and self._laser_scan_sequence
                <= self._last_laser_scan_query_sequence
            ):
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    return None
                self._laser_scan_condition.wait(timeout=remaining)
                if not self.is_connected:
                    return None

            self._last_laser_scan_query_sequence = self._laser_scan_sequence
            scan = self._latest_laser_scan
            assert scan is not None
            return LaserScan(
                angles=scan.angles.copy(),
                ranges=scan.ranges.copy(),
                intensities=scan.intensities.copy(),
                angle_min=scan.angle_min,
                angle_max=scan.angle_max,
                angle_increment=scan.angle_increment,
                time_increment=scan.time_increment,
                scan_time=scan.scan_time,
                range_min=scan.range_min,
                range_max=scan.range_max,
            )

    def query_image(
        self, timeout: Optional[float] = None, *, wait_for_new: bool = False
    ) -> Optional[np.ndarray]:
        """Return the latest camera image as a BGR NumPy array copy.

        Args:
            timeout: Seconds to wait for a suitable image. ``None`` waits
                indefinitely and ``0`` returns immediately.
            wait_for_new: Wait for an image received after the preceding query.

        Returns:
            The image, or ``None`` if the timeout expires or cleanup interrupts
            the query.
        """
        self._require_connection()
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._image_condition:
            while self._latest_image is None or (
                wait_for_new and self._image_sequence <= self._last_query_sequence
            ):
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    return None
                self._image_condition.wait(timeout=remaining)
                if not self.is_connected:
                    return None

            self._last_query_sequence = self._image_sequence
            return self._latest_image.copy()

    def cleanup(self) -> None:
        """Stop the robot, unsubscribe from topics, and disconnect safely."""
        if (
            self._client is not None
            and self._client.is_connected
            and "cmd_vel" in self._publishers
        ):
            try:
                self._publish_command(0.0, 0.0)
            except Exception as exc:
                print(f"Warning: could not send the stop command: {exc}")

        for topic in reversed(list(self._subscribers.values())):
            try:
                topic.unsubscribe()
            except Exception:
                pass

        for topic in reversed(list(self._publishers.values())):
            try:
                topic.unadvertise()
            except Exception:
                pass

        if self._client is not None:
            try:
                self._client.terminate()
            except Exception:
                pass

        self._subscribers.clear()
        self._publishers.clear()
        self._client = None
        self._is_setup = False
        with self._image_condition:
            self._latest_image = None
            self._image_condition.notify_all()
        with self._laser_scan_condition:
            self._latest_laser_scan = None
            self._laser_scan_condition.notify_all()

    def _add_publisher(
        self, name: str, topic_name: str, message_type: str
    ) -> roslibpy.Topic:
        """Create, register, and advertise a publisher."""
        if self._client is None:
            raise RuntimeError("The rosbridge client is not available")
        if name in self._publishers:
            raise ValueError(f"Publisher {name!r} is already registered")

        topic = roslibpy.Topic(self._client, topic_name, message_type)
        self._publishers[name] = topic
        topic.advertise()
        return topic

    def _add_subscriber(
        self,
        name: str,
        topic_name: str,
        message_type: str,
        callback: Callable[[dict], None],
    ) -> roslibpy.Topic:
        """Create, register, and subscribe a subscriber."""
        if self._client is None:
            raise RuntimeError("The rosbridge client is not available")
        if name in self._subscribers:
            raise ValueError(f"Subscriber {name!r} is already registered")

        topic = roslibpy.Topic(self._client, topic_name, message_type)
        self._subscribers[name] = topic
        topic.subscribe(callback)
        return topic

    def _publish_command(self, linear: float, angular: float) -> None:
        message = roslibpy.Message(
            {
                "linear": {"x": float(linear), "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": float(angular)},
            }
        )
        cmd_vel = self._publishers.get("cmd_vel")
        if cmd_vel is None:
            raise RuntimeError("The command topic is not available")
        cmd_vel.publish(message)

    def _require_connection(self) -> None:
        if not self.is_connected:
            raise RuntimeError("Robot is not connected. Call setup() first.")

    def _on_joint_state(self, message: dict) -> None:
        try:
            self._joint_states = message
        except (KeyError, TypeError, ValueError) as exc:
            print(f"Warning: could not decode joint state message: {exc}")

    def _on_image(self, message: dict) -> None:
        try:
            image_bytes = base64.b64decode(message["data"], validate=True)
            encoded_image = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
            if image is None:
                return
            with self._image_condition:
                self._latest_image = image
                self._image_sequence += 1
                self._image_condition.notify_all()
        except (KeyError, TypeError, ValueError) as exc:
            print(f"Warning: could not decode camera image: {exc}")

    def _on_laser_scan(self, message: dict) -> None:
        try:
            ranges = np.asarray(message["ranges"], dtype=np.float32)
            intensities = np.asarray(message.get("intensities", []), dtype=np.float32)
            if ranges.ndim != 1 or intensities.ndim != 1:
                raise ValueError("ranges and intensities must be one-dimensional")

            angle_min = float(message["angle_min"])
            angle_increment = float(message["angle_increment"])
            angles = (
                angle_min
                + np.arange(ranges.size, dtype=np.float32) * angle_increment
            )
            scan = LaserScan(
                angles=angles,
                ranges=ranges,
                intensities=intensities,
                angle_min=angle_min,
                angle_max=float(message["angle_max"]),
                angle_increment=angle_increment,
                time_increment=float(message.get("time_increment", 0.0)),
                scan_time=float(message.get("scan_time", 0.0)),
                range_min=float(message["range_min"]),
                range_max=float(message["range_max"]),
            )
            with self._laser_scan_condition:
                self._latest_laser_scan = scan
                self._laser_scan_sequence += 1
                self._laser_scan_condition.notify_all()
        except (KeyError, TypeError, ValueError) as exc:
            print(f"Warning: could not decode laser scan: {exc}")

    def __enter__(self) -> "RosbridgeRobot":
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.cleanup()
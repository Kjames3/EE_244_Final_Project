#!/usr/bin/env python3
"""
orbbec_depth_node.py — Publish Orbbec Astra Pro SC depth as ROS2 topics.

Publishes:
  /camera/depth/image_raw    (sensor_msgs/Image, encoding=16UC1, values in mm)
  /camera/depth/camera_info  (sensor_msgs/CameraInfo, Astra Pro defaults)

Usage (run on Jetson, after sourcing ROS2 + workspace):
  python3 ~/EE_244_Final_Project/robot/orbbec_depth_node.py

NOTE: Only one process can hold the OpenNI2/Orbbec device at a time.
      Do NOT run this alongside server_x3.py (which also opens the camera).
"""

import os
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

# Orbbec Astra Pro SC factory intrinsics at 640×480.
# Replace with calibrated values if you run camera_calibration.
ASTRA_FX = 570.3
ASTRA_FY = 570.3
ASTRA_CX = 319.5
ASTRA_CY = 239.5

# Paths to try when finding libOpenNI2.so.
# ~/x3_ws/src/ is first because that is where the Yahboom SDK ships the library
# and its OpenNI2/Drivers/ subdirectory on this Jetson.  The openni Python
# package also checks os.getcwd() implicitly, but an explicit path is safer.
_OPENNI2_SEARCH_DIRS = [
    os.path.expanduser("~/x3_ws/src"),   # Yahboom SDK location on Jetson
    None,                                 # openni default (cwd / OPENNI2_REDIST)
    "/usr/lib",
    "/usr/local/lib",
    "/opt/openni2/lib",
]


class OrbbecDepthNode(Node):

    def __init__(self):
        super().__init__('orbbec_depth_node')

        self._depth_stream = None
        self._oni_device   = None
        self._width        = 640
        self._height       = 480

        self._pub_depth = self.create_publisher(Image,      '/camera/depth/image_raw',   5)
        self._pub_info  = self.create_publisher(CameraInfo, '/camera/depth/camera_info', 5)

        self._init_openni2()

        if self._depth_stream is None:
            self.get_logger().fatal(
                'Could not open Orbbec depth stream. '
                'Checklist:\n'
                '  1. pip install openni\n'
                '  2. Camera USB cable is plugged in\n'
                '  3. server_x3.py is NOT running (it would hold the device)\n'
                '  4. Try: python3 -c "from openni import openni2; openni2.initialize(); print(openni2.Device.open_any())"'
            )
            sys.exit(1)

        self._info_msg = self._build_camera_info()

        # Publish at 30 Hz
        self.create_timer(1.0 / 30.0, self._publish_frame)
        self.get_logger().info(
            f'orbbec_depth_node: streaming /camera/depth/image_raw '
            f'({self._width}x{self._height} 16UC1) at 30 Hz'
        )

    # ── OpenNI2 init — mirrors AstraCamera._open_depth() in drivers_x3.py ───

    def _init_openni2(self):
        try:
            from openni import openni2

            initialized = False
            for search_dir in _OPENNI2_SEARCH_DIRS:
                try:
                    if search_dir is None:
                        openni2.initialize()
                    else:
                        openni2.initialize(search_dir)
                    initialized = True
                    self.get_logger().info(
                        f'OpenNI2 initialized '
                        f'({"system default" if search_dir is None else search_dir})'
                    )
                    break
                except Exception:
                    continue

            if not initialized:
                self.get_logger().error(
                    'OpenNI2 could not be initialized. '
                    'Ensure libOpenNI2.so is on LD_LIBRARY_PATH or in /usr/lib.'
                )
                return

            self._oni_device  = openni2.Device.open_any()
            stream = self._oni_device.create_depth_stream()
            stream.start()

            # Read actual resolution from the stream (don't assume 640×480)
            vm = stream.get_video_mode()
            self._width  = vm.resolutionX
            self._height = vm.resolutionY
            self._depth_stream = stream

        except ImportError:
            self.get_logger().error(
                'Python openni package not installed. Run: pip install openni'
            )
        except Exception as e:
            self.get_logger().error(f'OpenNI2 init error: {e}')

    # ── CameraInfo ───────────────────────────────────────────────────────────

    def _build_camera_info(self) -> CameraInfo:
        msg = CameraInfo()
        msg.header.frame_id = 'camera_link'
        msg.width  = self._width
        msg.height = self._height
        msg.distortion_model = 'plumb_bob'
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [
            ASTRA_FX, 0.0,      ASTRA_CX,
            0.0,      ASTRA_FY, ASTRA_CY,
            0.0,      0.0,      1.0,
        ]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [
            ASTRA_FX, 0.0,      ASTRA_CX, 0.0,
            0.0,      ASTRA_FY, ASTRA_CY, 0.0,
            0.0,      0.0,      1.0,      0.0,
        ]
        return msg

    # ── Publish loop ─────────────────────────────────────────────────────────

    def _publish_frame(self):
        if self._depth_stream is None:
            return
        try:
            frame = self._depth_stream.read_frame()
            buf   = frame.get_buffer_as_uint16()
            depth = np.frombuffer(buf, dtype=np.uint16).reshape(
                frame.height, frame.width
            )

            now = self.get_clock().now().to_msg()

            img = Image()
            img.header.stamp    = now
            img.header.frame_id = 'camera_link'
            img.height    = frame.height
            img.width     = frame.width
            img.encoding  = '16UC1'
            img.is_bigendian = False
            img.step      = frame.width * 2   # 2 bytes per pixel
            img.data      = depth.tobytes()
            self._pub_depth.publish(img)

            self._info_msg.header.stamp = now
            self._pub_info.publish(self._info_msg)

        except Exception as e:
            self.get_logger().warn(
                f'Depth frame read error: {e}',
                throttle_duration_sec=5.0,
            )

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def destroy_node(self):
        if self._depth_stream is not None:
            try:
                self._depth_stream.stop()
            except Exception:
                pass
        try:
            from openni import openni2
            openni2.unload()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = OrbbecDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

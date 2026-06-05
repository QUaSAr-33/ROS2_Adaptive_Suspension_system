import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from cv_bridge import CvBridge

import cv2
import numpy as np

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class FusionNode(Node):

    def __init__(self):
        super().__init__('fusion_node')

        self.declare_parameter('yolo_conf_threshold', 0.35)
        self.declare_parameter('high_motion_threshold', 0.25)
        self.declare_parameter('medium_motion_threshold', 0.05)
        self.declare_parameter('low_motion_threshold', 0.02)
        self.declare_parameter('motion_normalizer', 255.0)
        self.declare_parameter('enable_visualization', False)
        self.declare_parameter('fusion_rate', 25.0)

        self.yolo_conf_threshold = self.get_parameter('yolo_conf_threshold').value
        self.high_motion_threshold = self.get_parameter('high_motion_threshold').value
        self.medium_motion_threshold = self.get_parameter('medium_motion_threshold').value
        self.low_motion_threshold = self.get_parameter('low_motion_threshold').value
        self.motion_normalizer = self.get_parameter('motion_normalizer').value
        self.enable_vis = self.get_parameter('enable_visualization').value
        fusion_rate = self.get_parameter('fusion_rate').value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()

        self.yolo_cb_group = MutuallyExclusiveCallbackGroup()
        self.flow_cb_group = MutuallyExclusiveCallbackGroup()
        self.image_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()

        self.yolo_sub = self.create_subscription(
            Float32MultiArray, '/yolo/detections', self._yolo_callback, qos,
            callback_group=self.yolo_cb_group
        )
        self.flow_sub = self.create_subscription(
            Image, '/optical_flow/residual_map', self._flow_callback, qos,
            callback_group=self.flow_cb_group
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self._image_callback, qos,
            callback_group=self.image_cb_group
        )

        self.fusion_pub = self.create_publisher(Float32MultiArray, '/fusion/detections', qos)
        self.vis_pub = self.create_publisher(Image, '/fusion/visualization', qos)

        self.latest_detections = None
        self.latest_residual_map = None
        self.latest_frame = None
        self.latest_header = None
        self.roi_y_offset = 0

        self.fusion_busy = False
        self.frame_counter = 0
        self.total_fusion_time = 0.0
        self.last_fps_time = time.time()

        self.timer = self.create_timer(1.0 / fusion_rate, self._run_fusion,
                                       callback_group=self.timer_cb_group)

        self.get_logger().info(f"Fusion Node Started (Rate: {fusion_rate}Hz, Vis: {self.enable_vis})")

    def _yolo_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) == 0:
            self.latest_detections = np.zeros((0, 6), dtype=np.float32)
            return
        if len(data) % 6 != 0:
            self.get_logger().warning("Invalid YOLO format")
            return
        self.latest_detections = data.reshape(-1, 6)

    def _flow_callback(self, msg):
        try:
            self.latest_residual_map = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            try:
                self.roi_y_offset = int(msg.header.frame_id)
            except (ValueError, TypeError):
                self.roi_y_offset = 0
        except Exception as e:
            self.get_logger().warning(f"Residual map conversion failed: {e}")

    def _image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_header = msg.header
        except Exception as e:
            self.get_logger().warning(f"Frame conversion failed: {e}")

    def _run_fusion(self):
        if self.fusion_busy:
            return
        if self.latest_frame is None or self.latest_residual_map is None or self.latest_detections is None:
            return

        self.fusion_busy = True
        start_time = time.time()

        try:
            frame = self.latest_frame
            header = self.latest_header
            residual_map = self.latest_residual_map
            roi_offset = self.roi_y_offset
            detections = self.latest_detections

            fused_results = []
            roi_h, roi_w = residual_map.shape[:2]

            vis = frame.copy() if self.enable_vis else None

            for det in detections:
                x1, y1, x2, y2, yolo_conf, cls_id = det

                if yolo_conf < self.yolo_conf_threshold:
                    continue

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                roi_y1 = max(0, y1 - roi_offset)
                roi_y2 = min(roi_h - 1, y2 - roi_offset)
                roi_x1 = max(0, x1)
                roi_x2 = min(roi_w - 1, x2)

                motion_score = 0.0
                if roi_y2 > roi_y1 and roi_x2 > roi_x1:
                    roi_patch = residual_map[roi_y1:roi_y2, roi_x1:roi_x2]
                    if roi_patch.size > 0:
                        mean_motion = float(np.mean(roi_patch)) / self.motion_normalizer
                        variance_motion = float(np.var(roi_patch)) / (self.motion_normalizer ** 2)
                        high_motion_ratio = float(np.sum(roi_patch > 30) / roi_patch.size)
                        motion_score = min(
                            0.5 * mean_motion + 0.3 * variance_motion + 0.2 * high_motion_ratio,
                            1.0
                        )

                fused_results.extend([
                    float(x1), float(y1), float(x2), float(y2),
                    float(yolo_conf), float(motion_score)
                ])

                if self.enable_vis and vis is not None:
                    if motion_score > self.high_motion_threshold:
                        motion_label, color = "HIGH", (0, 255, 0)
                    elif motion_score > self.medium_motion_threshold:
                        motion_label, color = "MEDIUM", (0, 255, 255)
                    elif motion_score < self.low_motion_threshold:
                        motion_label, color = "LOW", (0, 165, 255)
                    else:
                        motion_label, color = "LOW-MED", (255, 255, 0)

                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    for i, text in enumerate([
                        f"Conf {yolo_conf:.2f}",
                        f"Motion {motion_label}",
                        f"Score {motion_score:.2f}",
                    ]):
                        cv2.putText(vis, text, (x1, y1 - 45 + i * 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

            fusion_msg = Float32MultiArray()
            fusion_msg.data = fused_results
            self.fusion_pub.publish(fusion_msg)

            if self.enable_vis and vis is not None:
                vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
                vis_msg.header = header
                self.vis_pub.publish(vis_msg)

            self.total_fusion_time += (time.time() - start_time)
            self.frame_counter += 1

            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            if elapsed >= 1.0:
                fps = self.frame_counter / elapsed
                avg_fusion_ms = (self.total_fusion_time / self.frame_counter) * 1000
                self.get_logger().info(
                    f"Fusion FPS: {fps:.1f} | Avg Execution: {avg_fusion_ms:.1f} ms | "
                    f"Fused Boxes: {len(fused_results)//6}"
                )
                self.frame_counter = 0
                self.total_fusion_time = 0.0
                self.last_fps_time = current_time

        except Exception as e:
            self.get_logger().error(f"Error in fusion loop: {e}")
        finally:
            self.fusion_busy = False


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
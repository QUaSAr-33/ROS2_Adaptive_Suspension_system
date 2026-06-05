import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int32

from cv_bridge import CvBridge

import cv2
import numpy as np

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class OpticalFlowNode(Node):

    def __init__(self):
        super().__init__('optical_flow')

        self.declare_parameter('pyr_scale', 0.5)
        self.declare_parameter('levels', 2)
        self.declare_parameter('winsize', 15)
        self.declare_parameter('iterations', 3)
        self.declare_parameter('poly_n', 5)
        self.declare_parameter('poly_sigma', 1.2)
        self.declare_parameter('normalize_scale', 10.0)
        self.declare_parameter('disturbance_threshold', 0.25)
        self.declare_parameter('flow_scale', 0.5)
        self.declare_parameter('flow_rate', 20.0)
        self.declare_parameter('enable_visualization', False)

        self.pyr_scale = self.get_parameter('pyr_scale').value
        self.levels = self.get_parameter('levels').value
        self.winsize = self.get_parameter('winsize').value
        self.iterations = self.get_parameter('iterations').value
        self.poly_n = self.get_parameter('poly_n').value
        self.poly_sigma = self.get_parameter('poly_sigma').value
        self.normalize_scale = self.get_parameter('normalize_scale').value
        self.disturbance_threshold = self.get_parameter('disturbance_threshold').value
        self.flow_scale = self.get_parameter('flow_scale').value
        self.flow_rate = self.get_parameter('flow_rate').value
        self.enable_vis = self.get_parameter('enable_visualization').value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()

        self.sub_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()

        self._latest_msg = None
        self.subscription = self.create_subscription(
            Image, '/camera/preprocessed', self._image_callback, qos,
            callback_group=self.sub_cb_group
        )

        self.roi_y_offset = 0
        self.offset_sub = self.create_subscription(
            Int32, '/camera/roi_y_offset', self._offset_callback, qos,
            callback_group=self.sub_cb_group
        )

        self.residual_pub = self.create_publisher(Image, '/optical_flow/residual_map', qos)
        self.stats_pub = self.create_publisher(Float32MultiArray, '/optical_flow/disturbance_stats', qos)
        self.heatmap_pub = self.create_publisher(Image, '/optical_flow/heatmap', qos)
        self.vis_pub = self.create_publisher(Image, '/optical_flow/vis', qos)

        self.prev_frame = None
        self._flow_busy = False

        self._small_prev = None
        self._last_fw = -1
        self._last_fh = -1
        self._scaled_w = -1
        self._scaled_h = -1

        self._stats_msg = Float32MultiArray()
        self._stats_msg.data = [0.0, 0.0, 0.0, 0.0]

        self.frame_counter = 0
        self.total_flow_time = 0.0
        self.last_fps_time = time.time()

        self.timer = self.create_timer(
            1.0 / self.flow_rate,
            self._run_flow,
            callback_group=self.timer_cb_group
        )

        self.get_logger().info(
            f"Optical Flow Node Started (Vis: {self.enable_vis}, Rate: {self.flow_rate}Hz, "
            f"FlowScale: {self.flow_scale})"
        )

    def _offset_callback(self, msg):
        self.roi_y_offset = msg.data

    def _image_callback(self, msg):
        self._latest_msg = msg

    def _run_flow(self):
        if self._flow_busy or self._latest_msg is None:
            return
        self._flow_busy = True
        msg = self._latest_msg
        self._latest_msg = None
        start_time = time.time()

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().warning(f"Image conversion failed: {e}")
            self._flow_busy = False
            return

        try:
            fh, fw = frame.shape[:2]

            if self.flow_scale < 1.0:
                if fw != self._last_fw or fh != self._last_fh:
                    self._scaled_w = int(fw * self.flow_scale)
                    self._scaled_h = int(fh * self.flow_scale)
                    self._last_fw = fw
                    self._last_fh = fh
                    self._small_prev = None

                small_curr = cv2.resize(frame, (self._scaled_w, self._scaled_h),
                                        interpolation=cv2.INTER_LINEAR)
            else:
                small_curr = frame

            if self._small_prev is None or self._small_prev.shape != small_curr.shape:
                self._small_prev = small_curr
                self.prev_frame = frame
                return

            flow = cv2.calcOpticalFlowFarneback(
                self._small_prev, small_curr, None,
                self.pyr_scale, self.levels, self.winsize,
                self.iterations, self.poly_n, self.poly_sigma, 0
            )

            self._small_prev = small_curr
            self.prev_frame = frame

            if self.flow_scale < 1.0:
                flow = cv2.resize(flow, (fw, fh), interpolation=cv2.INTER_LINEAR)
                flow *= (1.0 / self.flow_scale)

            flow_x = flow[..., 0]
            flow_y = flow[..., 1]

            global_dx = float(np.mean(flow_x))
            global_dy = float(np.mean(flow_y))

            residual_x = flow_x - global_dx
            residual_y = flow_y - global_dy

            residual_mag = cv2.magnitude(residual_x, residual_y)
            cv2.GaussianBlur(residual_mag, (3, 3), 0, dst=residual_mag)

            mean_disturbance = float(np.mean(residual_mag))
            max_disturbance = float(np.max(residual_mag))
            disturbance_variance = float(np.var(residual_mag))
            high_disturbance_ratio = float(
                np.sum(residual_mag > self.disturbance_threshold) / residual_mag.size
            )

            self._stats_msg.data = [
                mean_disturbance, max_disturbance,
                disturbance_variance, high_disturbance_ratio
            ]
            self.stats_pub.publish(self._stats_msg)

            np.clip(residual_mag * self.normalize_scale, 0, 255, out=residual_mag)
            residual_norm = residual_mag.astype(np.uint8)

            residual_msg = self.bridge.cv2_to_imgmsg(residual_norm, encoding='mono8')
            residual_msg.header = msg.header
            residual_msg.header.frame_id = str(self.roi_y_offset)
            self.residual_pub.publish(residual_msg)

            if self.enable_vis:
                heatmap = cv2.applyColorMap(residual_norm, cv2.COLORMAP_JET)
                heatmap_msg = self.bridge.cv2_to_imgmsg(heatmap, encoding='bgr8')
                heatmap_msg.header = msg.header
                self.heatmap_pub.publish(heatmap_msg)

                vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                overlay = cv2.addWeighted(vis, 0.6, heatmap, 0.4, 0)
                for i, (label, val) in enumerate([
                    ("Mean", mean_disturbance),
                    ("Max", max_disturbance),
                    ("Variance", disturbance_variance),
                    ("Ratio", high_disturbance_ratio),
                ]):
                    cv2.putText(overlay, f"{label}: {val:.3f}",
                                (20, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                vis_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
                vis_msg.header = msg.header
                self.vis_pub.publish(vis_msg)

            self.total_flow_time += (time.time() - start_time)
            self.frame_counter += 1

            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            if elapsed >= 1.0:
                fps = self.frame_counter / elapsed
                avg_exec_ms = (self.total_flow_time / self.frame_counter) * 1000
                self.get_logger().info(
                    f"Flow FPS: {fps:.1f} | Exec: {avg_exec_ms:.1f} ms | "
                    f"Mean Disturbance: {mean_disturbance:.3f}"
                )
                self.frame_counter = 0
                self.total_flow_time = 0.0
                self.last_fps_time = current_time

        except Exception as e:
            self.get_logger().error(f"Flow error: {e}")
        finally:
            self._flow_busy = False


def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowNode()
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
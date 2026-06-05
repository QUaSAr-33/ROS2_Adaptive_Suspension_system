import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from cv_bridge import CvBridge
import numpy as np

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy


class EstimatorTrack:
    def __init__(self, initial_dist, initial_depth, current_time):
        self.distance = initial_dist
        self.depth = initial_depth
        self.velocity = 0.0
        
        self.last_update_time = current_time
        self.hits = 1

    def update(self, raw_dist, raw_depth, conf, motion_score, current_time):
        dt = current_time - self.last_update_time
        if dt <= 0:
            return

        effective_conf = (0.8 * conf) + (0.2 * motion_score)

        alpha_d = np.clip(0.6 * effective_conf, 0.1, 0.85)
        alpha_v = np.clip(0.3 * effective_conf, 0.05, 0.4)
        alpha_depth = np.clip(0.4 * effective_conf, 0.1, 0.6)

        if self.hits == 1:
            raw_v = (self.distance - raw_dist) / dt
            self.velocity = np.clip(raw_v, -20.0, 20.0)
            self.distance = raw_dist
            self.depth = raw_depth
        else:
            raw_v = (self.distance - raw_dist) / dt
            raw_v = np.clip(raw_v, -20.0, 20.0)
            
            predicted_dist = self.distance - (self.velocity * dt)

            self.distance = (alpha_d * raw_dist) + ((1.0 - alpha_d) * predicted_dist)
            self.velocity = (alpha_v * raw_v) + ((1.0 - alpha_v) * self.velocity)
            self.depth = (alpha_depth * raw_depth) + ((1.0 - alpha_depth) * self.depth)

        self.last_update_time = current_time
        self.hits += 1

    def get_tti(self):
        if self.velocity <= 0.1:
            return 99.9
        return min(self.distance / self.velocity, 99.9)


class EstimationNode(Node):

    def __init__(self):
        super().__init__('estimation_node')

        self.declare_parameter('focal_length_px', 470.0)
        self.declare_parameter('image_height_px', 480.0)
        
        self.declare_parameter('assumed_pothole_width_m', 0.40)
        self.declare_parameter('camera_height_m', 0.20)  
        self.declare_parameter('width_blend_weight', 0.70)
        
        self.declare_parameter('depth_scale', 15.0)
        self.declare_parameter('min_distance_m', 0.5)
        self.declare_parameter('max_distance_m', 15.0)
        self.declare_parameter('track_timeout_s', 1.0)

        self.focal_length_px = self.get_parameter('focal_length_px').value
        self.image_height_px = self.get_parameter('image_height_px').value
        
        self.assumed_pothole_width_m = self.get_parameter('assumed_pothole_width_m').value
        self.camera_height_m = self.get_parameter('camera_height_m').value
        self.width_blend_weight = self.get_parameter('width_blend_weight').value
        
        self.depth_scale = self.get_parameter('depth_scale').value
        self.min_distance_m = self.get_parameter('min_distance_m').value
        self.max_distance_m = self.get_parameter('max_distance_m').value
        self.track_timeout_s = self.get_parameter('track_timeout_s').value

        self.optical_center_y = self.image_height_px / 2.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()

        self.track_cb_group = MutuallyExclusiveCallbackGroup()
        self.residual_cb_group = MutuallyExclusiveCallbackGroup()

        self.track_sub = self.create_subscription(
            Float32MultiArray, '/tracking/detections', self.track_callback, qos, callback_group=self.track_cb_group
        )

        self.residual_sub = self.create_subscription(
            Image, '/optical_flow/residual_map', self.residual_callback, qos, callback_group=self.residual_cb_group
        )

        self.result_pub = self.create_publisher(
            Float32MultiArray, '/estimation/results', qos
        )

        self.latest_residual_map = None
        self.roi_y_offset = 0
        self.active_tracks = {}

        self.frame_counter = 0
        self.total_est_time = 0.0
        self.last_fps_time = time.time()

        self.get_logger().info("Estimation V3 Node Started (Fused Distance Model)")

    def residual_callback(self, msg):
        try:
            self.latest_residual_map = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            try:
                self.roi_y_offset = int(msg.header.frame_id)
            except (ValueError, TypeError):
                self.roi_y_offset = 0
        except Exception as e:
            self.get_logger().warning(f"Residual conversion failed: {e}")

    def track_callback(self, msg):
        start_time = time.time()
        current_time = time.time()
        
        data = np.array(msg.data, dtype=np.float32)

        if len(data) == 0:
            self.result_pub.publish(Float32MultiArray())
            self._cleanup_tracks(current_time)
            self._log_performance(start_time, current_time)
            return

        if len(data) % 7 != 0:
            self.get_logger().warning("Invalid tracking format")
            return

        tracks = data.reshape(-1, 7)
        results = []
        current_track_ids = set()

        for track in tracks:
            x1, y1, x2, y2, conf, motion_score, track_id = track
            track_id = int(track_id)
            current_track_ids.add(track_id)

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            pixel_width = max(x2 - x1, 1)

            width_dist = (self.assumed_pothole_width_m * self.focal_length_px) / pixel_width
            
            bottom_y = float(y2)
            if bottom_y > self.optical_center_y:
                bottom_y_dist = (self.camera_height_m * self.focal_length_px) / (bottom_y - self.optical_center_y)
                raw_dist = (self.width_blend_weight * width_dist) + ((1.0 - self.width_blend_weight) * bottom_y_dist)
            else:
                raw_dist = width_dist
                
            raw_dist = float(np.clip(raw_dist, self.min_distance_m, self.max_distance_m))

            raw_depth = 0.0
            if self.latest_residual_map is not None:
                roi_h, roi_w = self.latest_residual_map.shape[:2]
                ry1 = int(max(0, y1 - self.roi_y_offset))
                ry2 = int(min(roi_h - 1, y2 - self.roi_y_offset))
                rx1 = int(max(0, x1))
                rx2 = int(min(roi_w - 1, x2))

                if ry2 > ry1 and rx2 > rx1:
                    patch = self.latest_residual_map[ry1:ry2, rx1:rx2].astype(np.float32)
                    if patch.size > 0:
                        variance = float(np.var(patch))
                        raw_depth = min(float(self.depth_scale * np.sqrt(variance)), 40.0)

            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = EstimatorTrack(raw_dist, raw_depth, current_time)
            else:
                self.active_tracks[track_id].update(raw_dist, raw_depth, conf, motion_score, current_time)

            est = self.active_tracks[track_id]
            tti = est.get_tti()

            results.extend([
                float(x1), float(y1),
                float(x2), float(y2),
                float(est.distance),
                float(est.depth),
                float(est.velocity),
                float(tti),
                float(track_id),
                float(conf),
                float(motion_score)
            ])

        out_msg = Float32MultiArray()
        out_msg.data = results
        self.result_pub.publish(out_msg)

        self._cleanup_tracks(current_time)
        self._log_performance(start_time, current_time)

    def _cleanup_tracks(self, current_time):
        stale_ids = [
            tid for tid, trk in self.active_tracks.items()
            if (current_time - trk.last_update_time) > self.track_timeout_s
        ]
        for tid in stale_ids:
            del self.active_tracks[tid]

    def _log_performance(self, start_time, current_time):
        self.total_est_time += (time.time() - start_time)
        self.frame_counter += 1
        
        elapsed = current_time - self.last_fps_time
        if elapsed >= 1.0:
            fps = self.frame_counter / elapsed
            avg_est_ms = (self.total_est_time / self.frame_counter) * 1000
            
            self.get_logger().info(
                f"Estimator FPS: {fps:.1f} | Avg Execution: {avg_est_ms:.2f} ms | Active Targets: {len(self.active_tracks)}"
            )

            self.frame_counter = 0
            self.total_est_time = 0.0
            self.last_fps_time = current_time


def main(args=None):
    rclpy.init(args=args)
    node = EstimationNode()
    
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
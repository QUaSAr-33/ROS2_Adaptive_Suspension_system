import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

from cv_bridge import CvBridge

import cv2
import numpy as np

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class VisualizationNode(Node):

    def __init__(self):
        super().__init__('visualization_node')

        self.declare_parameter('show_motion_score',    True)
        self.declare_parameter('show_track_id',        True)
        self.declare_parameter('show_confidence_bar',  True)
        self.declare_parameter('show_estimation',      True)
        self.declare_parameter('show_mid_zone',        True)
        self.declare_parameter('show_controller_hud',  True)
        self.declare_parameter('mid_zone_left',        0.30)
        self.declare_parameter('mid_zone_right',       0.70)
        self.declare_parameter('image_width',          640)
        self.declare_parameter('minimal_mode',         False)
        self.declare_parameter('vis_rate',             25.0)

        self.show_motion_score   = self.get_parameter('show_motion_score').value
        self.show_track_id       = self.get_parameter('show_track_id').value
        self.show_confidence_bar = self.get_parameter('show_confidence_bar').value
        self.show_estimation     = self.get_parameter('show_estimation').value
        self.show_mid_zone       = self.get_parameter('show_mid_zone').value
        self.show_controller_hud = self.get_parameter('show_controller_hud').value
        self.mid_zone_left       = self.get_parameter('mid_zone_left').value
        self.mid_zone_right      = self.get_parameter('mid_zone_right').value
        self.image_width         = self.get_parameter('image_width').value
        self.minimal_mode        = self.get_parameter('minimal_mode').value
        vis_rate                 = self.get_parameter('vis_rate').value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()

        self.img_cb_group    = MutuallyExclusiveCallbackGroup()
        self.track_cb_group  = MutuallyExclusiveCallbackGroup()
        self.est_cb_group    = MutuallyExclusiveCallbackGroup()
        self.ctrl_cb_group   = MutuallyExclusiveCallbackGroup()
        self.status_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group  = MutuallyExclusiveCallbackGroup()

        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self._image_callback, qos,
            callback_group=self.img_cb_group
        )
        self.track_sub = self.create_subscription(
            Float32MultiArray, '/tracking/detections', self._track_callback, qos,
            callback_group=self.track_cb_group
        )
        self.est_sub = self.create_subscription(
            Float32MultiArray, '/estimation/results', self._est_callback, qos,
            callback_group=self.est_cb_group
        )
        self.ctrl_sub = self.create_subscription(
            Float32MultiArray, '/controller/output', self._ctrl_callback, qos,
            callback_group=self.ctrl_cb_group
        )
        self.ctrl_status_sub = self.create_subscription(
            String, '/controller/status', self._status_callback, qos,
            callback_group=self.status_cb_group
        )

        self.vis_pub = self.create_publisher(Image, '/final/visualization', qos)

        self._latest_msg        = None
        self.latest_tracks      = []
        self.estimations        = {}
        self.controller_output  = [0.0, 0.0, 0.0, 0.0]
        self.controller_status  = ""

        self.vis_busy        = False
        self.frame_counter   = 0
        self.total_vis_time  = 0.0
        self.last_fps_time   = time.time()

        self._mid_zone_buf = None
        self._hud_buf      = None

        self.timer = self.create_timer(
            1.0 / vis_rate, self._run_visualization,
            callback_group=self.timer_cb_group
        )

        self.get_logger().info(f"Visualization Node Started (Rate: {vis_rate}Hz, Minimal: {self.minimal_mode})")

    def _image_callback(self, msg):
        self._latest_msg = msg

    def _track_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) == 0 or len(data) % 7 != 0:
            self.latest_tracks = []
            return
        self.latest_tracks = list(data.reshape(-1, 7))

    def _est_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) == 0 or len(data) % 11 != 0:
            return
        self.estimations = {}
        for row in data.reshape(-1, 11):
            tid = int(row[8])
            self.estimations[tid] = {
                'dist_m':  float(row[4]),
                'depth_cm':float(row[5]),
                'vel_mps': float(row[6]),
                'tti_s':   float(row[7]),
            }

    def _ctrl_callback(self, msg):
        if len(msg.data) >= 4:
            self.controller_output = list(msg.data[:4])

    def _status_callback(self, msg):
        self.controller_status = msg.data

    def _run_visualization(self):
        if self.vis_busy or self._latest_msg is None:
            return

        self.vis_busy = True
        msg = self._latest_msg
        start_time = time.time()

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warning(f"Frame conversion: {e}")
            self.vis_busy = False
            return

        try:
            vis = frame.copy()
            header = msg.header
            h, w = vis.shape[:2]

            mid_left_px  = int(w * self.mid_zone_left)
            mid_right_px = int(w * self.mid_zone_right)

            if self.show_mid_zone:
                roi = vis[0:h, mid_left_px:mid_right_px]
                if self._mid_zone_buf is None or self._mid_zone_buf.shape != roi.shape:
                    self._mid_zone_buf = np.full(roi.shape, 255, dtype=np.uint8)
                cv2.addWeighted(self._mid_zone_buf, 0.07, roi, 0.93, 0, roi)
                cv2.line(vis, (mid_left_px, 0),  (mid_left_px, h),  (200, 200, 200), 1)
                cv2.line(vis, (mid_right_px, 0), (mid_right_px, h), (200, 200, 200), 1)
                cv2.putText(vis, "ACTIVE ZONE", (mid_left_px + 5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            for track_arr in self.latest_tracks:
                x1, y1, x2, y2, conf, motion, track_id = track_arr
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id = int(track_id)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                in_zone = mid_left_px <= cx <= mid_right_px

                if in_zone:
                    color = (0, 255, 0) if conf > 0.85 else (0, 255, 255) if conf > 0.60 else (0, 165, 255)
                else:
                    color = (120, 120, 120)

                thickness = max(2, int(conf * 5))
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

                lines = []
                if self.minimal_mode:
                    label = f"ID:{track_id} {conf:.2f}" if self.show_track_id else f"Pothole {conf:.2f}"
                    lines.append(label)
                    if in_zone:
                        lines.append("[ ACTIVE ]")
                else:
                    lines.append(f"Pothole {conf:.2f}")
                    if self.show_track_id:
                        lines.append(f"ID {track_id}")
                    if self.show_motion_score:
                        m_label = "HIGH" if motion > 0.15 else "MED" if motion > 0.05 else "LOW"
                        lines.append(f"Motion {m_label}")
                    if self.show_estimation and track_id in self.estimations:
                        est = self.estimations[track_id]
                        lines.append(f"Dist: {est['dist_m']:.2f} m")
                        lines.append(f"Depth: {est['depth_cm']:.1f} cm")
                        lines.append(f"Vel: {est['vel_mps']:.1f} m/s")
                        if est['tti_s'] < 90.0:
                            lines.append(f"TTI: {est['tti_s']:.1f} s")
                        else:
                            lines.append("TTI: N/A")
                    if in_zone:
                        lines.append("[ IN ZONE ]")

                for i, text in enumerate(lines):
                    text_y = y1 - 10 - 22 * i
                    if text_y < 20:
                        text_y = y2 + 20 + 22 * i
                    cv2.putText(vis, text, (x1, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                if self.show_confidence_bar:
                    bar_w = int((x2 - x1) * conf)
                    cv2.rectangle(vis, (x1, y2 + 6), (x2, y2 + 16), (60, 60, 60), -1)
                    cv2.rectangle(vis, (x1, y2 + 6), (x1 + bar_w, y2 + 16), color, -1)

            if self.show_controller_hud:
                pwm, depth_mm, dist_cm, in_zone_f = self.controller_output
                in_zone_active = in_zone_f > 0.5

                hud_color = (0, 255, 128) if in_zone_active else (160, 160, 160)
                hud_lines = [
                    f"Tracked: {len(self.latest_tracks)}",
                    f"PWM:     {int(pwm):3d} / 255",
                    f"Dist:    {dist_cm/100:.2f} m",
                    f"Depth:   {depth_mm/10:.1f} cm",
                    f"PID:     {'ACTIVE' if in_zone_active else 'IDLE'}",
                ]

                panel_x, panel_y = 10, 30
                panel_h = len(hud_lines) * 26 + 12
                panel_w = 200

                if panel_y + panel_h <= h and panel_x + panel_w <= w:
                    roi_hud = vis[panel_y - 22:panel_y + panel_h - 22,
                                  panel_x - 5:panel_x + panel_w - 5]
                    if self._hud_buf is None or self._hud_buf.shape != roi_hud.shape:
                        self._hud_buf = np.full(roi_hud.shape, 20, dtype=np.uint8)
                    cv2.addWeighted(self._hud_buf, 0.55, roi_hud, 0.45, 0, roi_hud)

                for i, line in enumerate(hud_lines):
                    cv2.putText(vis, line, (panel_x, panel_y + i * 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)

            vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            vis_msg.header = header
            self.vis_pub.publish(vis_msg)

            self.total_vis_time += (time.time() - start_time)
            self.frame_counter  += 1

            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            if elapsed >= 1.0:
                fps         = self.frame_counter / elapsed
                avg_exec_ms = (self.total_vis_time / self.frame_counter) * 1000
                self.get_logger().info(
                    f"Vis FPS: {fps:.1f} | Exec: {avg_exec_ms:.1f} ms | "
                    f"Tracks Drawn: {len(self.latest_tracks)}"
                )
                self.frame_counter  = 0
                self.total_vis_time = 0.0
                self.last_fps_time  = current_time

        except Exception as e:
            self.get_logger().error(f"Error in visualization loop: {e}")
        finally:
            self.vis_busy = False


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
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
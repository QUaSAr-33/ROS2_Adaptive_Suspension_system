import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import Float32MultiArray, String

import numpy as np
import serial
import time

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


_PHASE_IDLE     = 'IDLE'
_PHASE_APPROACH = 'APPROACH'
_PHASE_IMPACT   = 'IMPACT'
_PHASE_HOLD     = 'HOLD'


class ControllerNode(Node):

    def __init__(self):
        super().__init__('controller_node')

        self.declare_parameter('bottom_y_trigger_frac', 0.70)
        self.declare_parameter('bottom_y_impact_frac',  0.92)

        self.declare_parameter('impact_hold_sec',   0.40)
        self.declare_parameter('actuator_lag_sec',  0.20)

        self.declare_parameter('max_pwm',     200)
        self.declare_parameter('idle_pwm',      0)
        self.declare_parameter('preload_pwm',  80)

        self.declare_parameter('mid_zone_left',  0.30)
        self.declare_parameter('mid_zone_right', 0.70)
        self.declare_parameter('image_width',   640)
        self.declare_parameter('image_height',  480)

        self.declare_parameter('serial_port',    '/dev/ttyUSB0')
        self.declare_parameter('baud_rate',      115200)
        self.declare_parameter('serial_timeout', 0.05)

        self.declare_parameter('min_conf_to_act', 0.45)
        self.declare_parameter('control_rate',    20.0)

        self.bottom_y_trigger_frac = self.get_parameter('bottom_y_trigger_frac').value
        self.bottom_y_impact_frac  = self.get_parameter('bottom_y_impact_frac').value
        self.impact_hold_sec       = self.get_parameter('impact_hold_sec').value
        self.actuator_lag_sec      = self.get_parameter('actuator_lag_sec').value
        self.max_pwm               = self.get_parameter('max_pwm').value
        self.idle_pwm              = self.get_parameter('idle_pwm').value
        self.preload_pwm           = self.get_parameter('preload_pwm').value
        self.mid_zone_left         = self.get_parameter('mid_zone_left').value
        self.mid_zone_right        = self.get_parameter('mid_zone_right').value
        self.image_width           = self.get_parameter('image_width').value
        self.image_height          = self.get_parameter('image_height').value
        self.serial_port           = self.get_parameter('serial_port').value
        self.baud_rate             = self.get_parameter('baud_rate').value
        self.serial_timeout        = self.get_parameter('serial_timeout').value
        self.min_conf_to_act       = self.get_parameter('min_conf_to_act').value
        self.control_rate          = self.get_parameter('control_rate').value

        self.mid_px_left  = int(self.image_width * self.mid_zone_left)
        self.mid_px_right = int(self.image_width * self.mid_zone_right)

        self._phase_to_pwm = {
            _PHASE_IDLE:     self.idle_pwm,
            _PHASE_APPROACH: self.preload_pwm,
            _PHASE_IMPACT:   self.max_pwm,
            _PHASE_HOLD:     self.preload_pwm,
        }

        self.ser = None
        self._init_serial()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.est_cb_group   = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()

        self.est_sub = self.create_subscription(
            Float32MultiArray, '/estimation/results',
            self._estimation_callback, qos,
            callback_group=self.est_cb_group
        )

        self.output_pub = self.create_publisher(Float32MultiArray, '/controller/output', qos)
        self.status_pub = self.create_publisher(String, '/controller/status', qos)

        self.latest_estimations = {}

        self._phase           = _PHASE_IDLE
        self._active_tid      = -1
        self._active_depth_cm = 0.0
        self._active_dist_m   = 0.0
        self._active_y        = 0.0
        self._active_tti      = float('inf')

        self._latched_pwm     = self.idle_pwm
        self._hold_until      = 0.0

        self._last_sent_phase = None

        self.loop_counter      = 0
        self.total_loop_time   = 0.0
        self.last_fps_time     = time.time()

        self._out_msg    = Float32MultiArray()
        self._status_msg = String()

        self.get_logger().info(
            f"Controller Node Started | QoS Depth: 1 | Rate: {self.control_rate}Hz"
        )

        self.timer = self.create_timer(
            1.0 / self.control_rate,
            self._control_loop,
            callback_group=self.timer_cb_group
        )

    def _init_serial(self):
        try:
            self.ser = serial.Serial(
                self.serial_port, self.baud_rate, timeout=self.serial_timeout
            )
            time.sleep(2.0)
            self.get_logger().info(f"Serial opened: {self.serial_port} @ {self.baud_rate}")
        except serial.SerialException as e:
            self.get_logger().warning(f"Serial unavailable: {e} (dry-run mode)")
            self.ser = None

    def _send_serial(self, pwm: int, depth_mm: int, dist_cm: int):
        if self.ser is None or not self.ser.is_open:
            return
        try:
            self.ser.write(f"S{pwm:03d},{depth_mm:04d},{dist_cm:04d}\n".encode('ascii'))
        except serial.SerialException as e:
            self.get_logger().warning(f"Serial write failed: {e}")
            self.ser = None

    def _estimation_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) == 0:
            self.latest_estimations = {}
            return
        if len(data) % 11 != 0:
            self.get_logger().warning("Invalid estimation format.")
            return
        self.latest_estimations = {}
        for row in data.reshape(-1, 11):
            tid = int(row[8])
            yf  = float(row[3]) / self.image_height
            self.latest_estimations[tid] = {
                'x1':          float(row[0]),
                'y1':          float(row[1]),
                'x2':          float(row[2]),
                'y2':          float(row[3]),
                'dist_m':      float(row[4]),
                'depth_cm':    float(row[5]),
                'velocity':    float(row[6]),
                'tti_sec':     float(row[7]),
                'conf':        float(row[9]),
                'motion_score':float(row[10]),
                'bottom_y_frac': yf,
            }

    def _best_candidate(self):
        best = None
        best_score = -1.0
        for tid, est in self.latest_estimations.items():
            if est['conf'] < self.min_conf_to_act:
                continue
            cx = (est['x1'] + est['x2']) / 2.0
            if not (self.mid_px_left <= cx <= self.mid_px_right):
                continue
            tti = est['tti_sec']
            bottom_y = est['bottom_y_frac']
            if tti > 1.5 and bottom_y < self.bottom_y_trigger_frac:
                continue
            norm_depth = min(est['depth_cm'] / 40.0, 1.0)
            urgency    = min(1.0 / max(tti, 0.05), 4.0)
            score      = (0.6 * norm_depth * est['conf']) + (0.4 * urgency)
            if score > best_score:
                best_score = score
                best = (tid, est)
        return best

    def _control_loop(self):
        start_time = time.time()
        now_wall   = time.monotonic()

        candidate = self._best_candidate()

        prev_phase = self._phase

        if self._phase == _PHASE_IDLE:
            if candidate is not None:
                tid, est = candidate
                self._active_tid      = tid
                self._active_depth_cm = est['depth_cm']
                self._active_dist_m   = est['dist_m']
                self._active_y        = est['bottom_y_frac']
                self._active_tti      = est['tti_sec']
                self._phase           = _PHASE_APPROACH

        elif self._phase == _PHASE_APPROACH:
            still_here = self._active_tid in self.latest_estimations
            if not still_here:
                self._phase      = _PHASE_IDLE
                self._active_tid = -1
            else:
                est = self.latest_estimations[self._active_tid]
                self._active_depth_cm = est['depth_cm']
                self._active_dist_m   = est['dist_m']
                self._active_y        = est['bottom_y_frac']
                self._active_tti      = est['tti_sec']

                if (est['tti_sec'] <= self.actuator_lag_sec or
                        est['bottom_y_frac'] >= self.bottom_y_impact_frac):
                    self._phase      = _PHASE_IMPACT
                    self._hold_until = now_wall + self.impact_hold_sec

        elif self._phase == _PHASE_IMPACT:
            still_here = self._active_tid in self.latest_estimations
            if still_here:
                est = self.latest_estimations[self._active_tid]
                self._active_depth_cm = est['depth_cm']
                self._active_dist_m   = est['dist_m']
                self._active_y        = est['bottom_y_frac']
                self._active_tti      = est['tti_sec']

            if now_wall >= self._hold_until:
                self._phase = _PHASE_HOLD

        elif self._phase == _PHASE_HOLD:
            if now_wall >= self._hold_until + self.impact_hold_sec:
                self._phase      = _PHASE_IDLE
                self._active_tid = -1

        self._latched_pwm = self._phase_to_pwm[self._phase]

        phase_changed = (self._phase != prev_phase)
        if phase_changed:
            depth_mm = int(round(self._active_depth_cm * 10))
            dist_cm  = int(round(self._active_dist_m * 100))
            self._send_serial(self._latched_pwm, depth_mm, dist_cm)
            self._last_sent_phase = self._phase

        self._out_msg.data = [
            float(self._latched_pwm),
            float(int(round(self._active_depth_cm * 10))),
            float(int(round(self._active_dist_m * 100))),
            float(self._phase in (_PHASE_APPROACH, _PHASE_IMPACT, _PHASE_HOLD)),
        ]
        self.output_pub.publish(self._out_msg)

        tti_str    = f"{self._active_tti:.2f}s" if self._active_tti != float('inf') else "inf"
        status_str = (
            f"Phase={self._phase:<8s} PWM={self._latched_pwm:3d} "
            f"dist={self._active_dist_m:.2f}m depth={self._active_depth_cm:.1f}cm "
            f"bottom_y={self._active_y:.2f} tti={tti_str} track={self._active_tid}"
        )
        self._status_msg.data = status_str
        self.status_pub.publish(self._status_msg)

        self.total_loop_time += (time.time() - start_time)
        self.loop_counter    += 1

        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        if elapsed >= 1.0:
            fps         = self.loop_counter / elapsed
            avg_exec_ms = (self.total_loop_time / self.loop_counter) * 1000
            self.get_logger().info(
                f"Control FPS: {fps:.1f} | Exec: {avg_exec_ms:.1f} ms | Phase: {self._phase:<8s} | "
                f"Target: {self._active_tid} | PWM: {self._latched_pwm}"
            )
            self.get_logger().info(status_str)
            self.loop_counter    = 0
            self.total_loop_time = 0.0
            self.last_fps_time   = current_time

    def destroy_node(self):
        if self.ser and self.ser.is_open:
            self._send_serial(self.idle_pwm, 0, 0)
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
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
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class PreprocessingNode(Node):
    def __init__(self):
        super().__init__('preprocessing_node')

        self.declare_parameter('roi_ratio', 0.5)
        self.declare_parameter('clahe_clip', 2.0)
        self.declare_parameter('blur_kernel', 5)
        self.declare_parameter('enable_blur', True)
        self.declare_parameter('process_rate', 25.0)

        self.roi_ratio = self.get_parameter('roi_ratio').value
        self.clahe_clip = self.get_parameter('clahe_clip').value
        self.blur_kernel = self.get_parameter('blur_kernel').value
        self.enable_blur = self.get_parameter('enable_blur').value
        self.process_rate = self.get_parameter('process_rate').value

        k = self.blur_kernel
        if k % 2 == 0:
            k += 1
        self._blur_k = k
        self._blur_ksize = (k, k)

        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(8, 8)
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()
        self.sub_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()

        self._latest_msg = None
        self._proc_busy = False
        self._cached_roi_start = -1
        self._cached_height = -1

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self._image_callback,
            qos,
            callback_group=self.sub_cb_group
        )

        self.publisher = self.create_publisher(Image, '/camera/preprocessed', qos)
        self.offset_pub = self.create_publisher(Int32, '/camera/roi_y_offset', qos)

        self._offset_msg = Int32()

        self.timer = self.create_timer(
            1.0 / self.process_rate,
            self._process_tick,
            callback_group=self.timer_cb_group
        )

        self.get_logger().info("Preprocessing Node Started")

    def _image_callback(self, msg):
        self._latest_msg = msg

    def _process_tick(self):
        if self._proc_busy or self._latest_msg is None:
            return
        self._proc_busy = True
        msg = self._latest_msg
        self._latest_msg = None
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warning(f"Image conversion failed: {e}")
            self._proc_busy = False
            return

        try:
            h, w = frame.shape[:2]

            if h != self._cached_height:
                self._cached_roi_start = int(h * self.roi_ratio)
                self._cached_height = h
                self._offset_msg.data = self._cached_roi_start

            roi_start = self._cached_roi_start
            roi = frame[roi_start:h, :]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = self._clahe.apply(gray)
            if self.enable_blur:
                gray = cv2.GaussianBlur(gray, self._blur_ksize, 0)

            out_msg = self.bridge.cv2_to_imgmsg(gray, encoding='mono8')
            out_msg.header = msg.header
            self.publisher.publish(out_msg)
            self.offset_pub.publish(self._offset_msg)
        finally:
            self._proc_busy = False


def main(args=None):
    rclpy.init(args=args)
    node = PreprocessingNode()
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
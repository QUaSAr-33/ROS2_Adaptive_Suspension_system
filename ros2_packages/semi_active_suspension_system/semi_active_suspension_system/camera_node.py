import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('device_id', 0)
        self.declare_parameter('fps', 30)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('video_path', '')
        self.declare_parameter('loop_video', True)

        self.device_id = self.get_parameter('device_id').value
        self.fps = self.get_parameter('fps').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.video_path = self.get_parameter('video_path').value
        self.loop_video = self.get_parameter('loop_video').value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.publisher = self.create_publisher(Image, '/camera/image_raw', qos)
        self.bridge = CvBridge()

        source = self.video_path if self.video_path else self.device_id
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open capture source: {source}")
            raise RuntimeError("Camera/video failed to open")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.video_path:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._capture_busy = False
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(
            1.0 / self.fps,
            self.capture_frame,
            callback_group=self.timer_cb_group
        )

        self.get_logger().info(
            f"Camera Node Started  |  source={source}  |  fps={self.fps}"
        )

    def capture_frame(self):
        if self._capture_busy:
            return
        self._capture_busy = True
        try:
            ret, frame = self.cap.read()

            if not ret:
                if self.video_path and self.loop_video:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        self.get_logger().warning("Loop: could not read frame after reset")
                        return
                else:
                    self.get_logger().warning("Frame capture failed")
                    return

            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height),
                                   interpolation=cv2.INTER_LINEAR)

            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(msg)
        finally:
            self._capture_busy = False

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
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
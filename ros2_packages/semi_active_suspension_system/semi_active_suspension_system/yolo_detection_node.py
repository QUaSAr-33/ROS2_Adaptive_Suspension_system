import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class YOLODetectionNode(Node):

    def __init__(self):
        super().__init__('yolo_detection_node')

        self.declare_parameter(
            'model_path',
            '/home/admirer/workspace/deeplearning/machine_learning/projects/'
            'adaptive_suspension/scripts/runs/detect/train/weights/best.pt'
        )
        self.declare_parameter('confidence_threshold', 0.35)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('half', True)
        self.declare_parameter('enable_visualization', False)
        self.declare_parameter('inference_rate', 10.0)
        self.declare_parameter('skip_frames', 0)

        self.model_path     = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.image_size     = self.get_parameter('image_size').value
        self.device         = self.get_parameter('device').value
        self.half           = self.get_parameter('half').value
        self.enable_vis     = self.get_parameter('enable_visualization').value
        inference_rate      = self.get_parameter('inference_rate').value
        self.skip_frames    = self.get_parameter('skip_frames').value

        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            self.half   = False

        try:
            self.model = YOLO(self.model_path)
            if self.half and self.device != 'cpu':
                self.model.model.half()
            dummy_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            self.model.predict(
                source=dummy_img, imgsz=self.image_size,
                device=self.device, half=self.half, verbose=False
            )
        except Exception as e:
            raise RuntimeError("YOLO model loading failed")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()

        self._latest_frame  = None
        self._latest_header = None
        self._skip_count    = 0

        self.inference_busy    = False
        self.frame_counter     = 0
        self.total_infer_time  = 0.0
        self.last_fps_time     = time.time()

        self._det_msg = Float32MultiArray()

        self.sub_cb_group   = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self._image_callback, qos,
            callback_group=self.sub_cb_group
        )

        self.detection_pub = self.create_publisher(Float32MultiArray, '/yolo/detections', qos)
        self.vis_pub       = self.create_publisher(Image, '/yolo/visualization', qos)

        self.timer = self.create_timer(
            1.0 / inference_rate, self._run_yolo,
            callback_group=self.timer_cb_group
        )

    def _image_callback(self, msg):
        self._latest_frame = msg

    def _run_yolo(self):
        if self.inference_busy or self._latest_frame is None:
            return

        if self.skip_frames > 0:
            self._skip_count += 1
            if self._skip_count <= self.skip_frames:
                return
            self._skip_count = 0

        self.inference_busy = True
        msg    = self._latest_frame
        header = msg.header

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            self.inference_busy = False
            return

        try:
            start_time = time.time()

            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                imgsz=self.image_size,
                device=self.device,
                half=self.half,
                verbose=False
            )

            infer_time = time.time() - start_time
            self.total_infer_time += infer_time
            self.frame_counter    += 1

            detections     = []
            num_detections = 0
            vis = frame.copy() if self.enable_vis else None

            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                box_data = result.boxes.data.cpu().numpy()
                num_detections += len(box_data)
                for row in box_data:
                    x1, y1, x2, y2, conf, cls_id = row
                    detections.extend([
                        float(x1), float(y1), float(x2), float(y2),
                        float(conf), float(cls_id)
                    ])
                    if self.enable_vis and vis is not None:
                        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(vis, f"Pothole {conf:.2f}",
                                    (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

            self._det_msg.data = detections
            self.detection_pub.publish(self._det_msg)

            if self.enable_vis and vis is not None:
                vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
                vis_msg.header = header
                self.vis_pub.publish(vis_msg)

            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            if elapsed >= 1.0:
                fps           = self.frame_counter / elapsed
                avg_infer_ms  = (self.total_infer_time / self.frame_counter) * 1000
                msg_time_sec  = header.stamp.sec + (header.stamp.nanosec * 1e-9)
                now_sec       = self.get_clock().now().nanoseconds * 1e-9
                latency_ms    = (now_sec - msg_time_sec) * 1000
                self.get_logger().info(
                    f"YOLO FPS: {fps:.1f} | Inference: {avg_infer_ms:.1f} ms | "
                    f"Latency: {latency_ms:.1f} ms | Detections: {num_detections}"
                )
                self.frame_counter    = 0
                self.total_infer_time = 0.0
                self.last_fps_time    = current_time

        except Exception:
            pass
        finally:
            self.inference_busy = False


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectionNode()
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
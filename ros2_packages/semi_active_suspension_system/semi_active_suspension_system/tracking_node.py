import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    o = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    t = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    iou = wh / (o + t - wh + 1e-16)
    return iou


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox, conf, motion_score):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.x = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)

        self.F = np.eye(8, dtype=np.float32)

        self.H = np.eye(4, 8, dtype=np.float32)

        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0 

        self.R = np.eye(4, dtype=np.float32) * 1.0

        self.Q = np.eye(8, dtype=np.float32) * 0.5
        self.Q[4:, 4:] *= 0.01

        self.time_since_update = 0
        self.hits = 1
        self.conf = conf
        self.motion_score = motion_score

    def predict(self, dt):
        for i in range(4):
            self.F[i, i + 4] = dt

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox, conf, motion_score, dt):
        self.time_since_update = 0
        
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        if self.hits == 1 and dt > 0:
            self.x[4, 0] = (cx - self.x[0, 0]) / dt
            self.x[5, 0] = (cy - self.x[1, 0]) / dt
            self.x[6, 0] = (w - self.x[2, 0]) / dt
            self.x[7, 0] = (h - self.x[3, 0]) / dt

        self.hits += 1
        
        self.conf = 0.8 * self.conf + 0.2 * conf
        self.motion_score = 0.8 * self.motion_score + 0.2 * motion_score

        Z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)

        Y = Z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, Y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_state(self):
        cx, cy, w, h = self.x[:4, 0]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)


class TrackingNode(Node):

    def __init__(self):
        super().__init__('tracking_node')

        self.declare_parameter('max_missing_frames', 5)
        self.declare_parameter('min_hits', 3)
        self.declare_parameter('iou_threshold', 0.2)

        self.max_missing_frames = self.get_parameter('max_missing_frames').value
        self.min_hits = self.get_parameter('min_hits').value
        self.iou_threshold = self.get_parameter('iou_threshold').value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_cb_group = MutuallyExclusiveCallbackGroup()

        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/fusion/detections',
            self.callback,
            qos,
            callback_group=self.sub_cb_group
        )

        self.publisher = self.create_publisher(
            Float32MultiArray,
            '/tracking/detections',
            qos
        )

        self.tracks = []
        KalmanBoxTracker.count = 0 
        
        self.frame_counter = 0
        self.total_tracking_time = 0.0
        self.last_fps_time = time.time()
        
        self.last_update_time = time.time()

        self.get_logger().info("Hungarian Kalman Tracking Node Started")

    def callback(self, msg):
        start_time = time.time()
        
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        dt = float(np.clip(dt, 0.01, 0.15))

        data = np.array(msg.data, dtype=np.float32)

        if len(data) == 0:
            self._update_empty(dt)
            self._publish_tracks()
            
            self.total_tracking_time += (time.time() - start_time)
            self.frame_counter += 1
            self._log_performance()
            return

        if len(data) % 6 != 0:
            self.get_logger().warning("Invalid fusion detection format")
            return

        detections = data.reshape(-1, 6)
        det_boxes = detections[:, :4]
        det_confs = detections[:, 4]
        det_motion = detections[:, 5]

        trk_boxes = []
        for trk in self.tracks:
            trk_boxes.append(trk.predict(dt))
        
        trk_boxes = np.array(trk_boxes) if len(trk_boxes) > 0 else np.empty((0, 4))

        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            det_boxes, trk_boxes, self.iou_threshold
        )

        for m in matched:
            det_idx, trk_idx = m[0], m[1]
            self.tracks[trk_idx].update(
                det_boxes[det_idx], 
                det_confs[det_idx], 
                det_motion[det_idx],
                dt
            )

        for i in unmatched_dets:
            trk = KalmanBoxTracker(det_boxes[i], det_confs[i], det_motion[i])
            self.tracks.append(trk)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_missing_frames]

        self._publish_tracks()
        
        self.total_tracking_time += (time.time() - start_time)
        self.frame_counter += 1
        self._log_performance()

    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
            
        iou_matrix = iou_batch(detections, trackers)
        
        cost_matrix = 1.0 - iou_matrix
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_indices = []
        unmatched_detections = []
        unmatched_trackers = []

        for d, t in zip(row_ind, col_ind):
            if iou_matrix[d, t] < iou_threshold:
                unmatched_detections.append(d)
                unmatched_trackers.append(t)
            else:
                matched_indices.append([d, t])

        unmatched_detections.extend([d for d in range(len(detections)) if d not in row_ind])
        unmatched_trackers.extend([t for t in range(len(trackers)) if t not in col_ind])

        return np.array(matched_indices), np.array(unmatched_detections), np.array(unmatched_trackers)

    def _update_empty(self, dt):
        for trk in self.tracks:
            trk.predict(dt)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_missing_frames]

    def _publish_tracks(self):
        results = []
        for trk in self.tracks:
            if trk.hits >= self.min_hits and trk.time_since_update <= self.max_missing_frames:
                box = trk.get_state()
                results.extend([
                    float(box[0]), float(box[1]),
                    float(box[2]), float(box[3]),
                    float(trk.conf),
                    float(trk.motion_score),
                    float(trk.id)
                ])

        out_msg = Float32MultiArray()
        out_msg.data = results
        self.publisher.publish(out_msg)

    def _log_performance(self):
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        if elapsed >= 1.0:
            fps = self.frame_counter / elapsed
            avg_tracking_ms = (self.total_tracking_time / self.frame_counter) * 1000 if self.frame_counter > 0 else 0
            
            self.get_logger().info(
                f"Tracker FPS: {fps:.1f} | Avg Execution: {avg_tracking_ms:.2f} ms | Active Tracks: {len(self.tracks)}"
            )

            self.frame_counter = 0
            self.total_tracking_time = 0.0
            self.last_fps_time = current_time


def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    
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
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    args = [
        DeclareLaunchArgument('video_path',       default_value='',             description='Path to video file (empty = use camera)'),
        DeclareLaunchArgument('device_id',        default_value='0',            description='Camera device ID'),
        DeclareLaunchArgument('fps',              default_value='30',           description='Camera FPS'),
        DeclareLaunchArgument('image_width',      default_value='640',          description='Frame width (px)'),
        DeclareLaunchArgument('image_height',     default_value='480',          description='Frame height (px)'),
        DeclareLaunchArgument('model_path',       default_value='/home/admirer/workspace/deeplearning/machine_learning/projects/adaptive_suspension/scripts/runs/detect/train/weights/best.pt',
                                                                                description='Path to YOLO .pt weights'),
        DeclareLaunchArgument('yolo_device',      default_value='cuda',         description='YOLO device: cuda or cpu'),
        DeclareLaunchArgument('yolo_conf',        default_value='0.35',         description='YOLO confidence threshold'),
        DeclareLaunchArgument('serial_port',      default_value='/dev/ttyUSB0', description='Arduino/ESP32 serial port'),
        DeclareLaunchArgument('baud_rate',        default_value='115200',       description='Serial baud rate'),
        DeclareLaunchArgument('mid_zone_left',    default_value='0.30',         description='Mid-zone left boundary (fraction)'),
        DeclareLaunchArgument('mid_zone_right',   default_value='0.70',         description='Mid-zone right boundary (fraction)'),
        DeclareLaunchArgument('focal_length_px',  default_value='470.0',        description='Camera focal length in pixels'),
    ]

    video_path      = LaunchConfiguration('video_path')
    device_id       = LaunchConfiguration('device_id')
    fps             = LaunchConfiguration('fps')
    image_width     = LaunchConfiguration('image_width')
    image_height    = LaunchConfiguration('image_height')
    model_path      = LaunchConfiguration('model_path')
    yolo_device     = LaunchConfiguration('yolo_device')
    yolo_conf       = LaunchConfiguration('yolo_conf')
    serial_port     = LaunchConfiguration('serial_port')
    baud_rate       = LaunchConfiguration('baud_rate')
    mid_zone_left   = LaunchConfiguration('mid_zone_left')
    mid_zone_right  = LaunchConfiguration('mid_zone_right')
    focal_length_px = LaunchConfiguration('focal_length_px')

    camera_node = Node(
        package='semi_active_suspension_system',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{
            'video_path':  video_path,
            'device_id':   device_id,
            'fps':         fps,
            'width':       image_width,
            'height':      image_height,
            'loop_video':  True,
        }]
    )
    preprocessing_node = Node(
        package='semi_active_suspension_system',
        executable='preprocessing_node',
        name='preprocessing_node',
        output='screen',

    )

    yolo_node = Node(
        package='semi_active_suspension_system',
        executable='yolo_detection_node',
        name='yolo_detection_node',
        output='screen',
        parameters=[{
            'model_path':            model_path,
            'confidence_threshold':  yolo_conf,
            'image_size':            640,
            'device':                yolo_device,
            'half':                  True,
            'skip_frames':           0,
            'inference_rate':        10.0,
        }]
    )
    optical_flow_node = Node(
        package='semi_active_suspension_system',
        executable='optical_flow',
        name='optical_flow',
        output='screen',

    )
    

    fusion_node = Node(
        package='semi_active_suspension_system',
        executable='fusion_node',
        name='fusion_node',
        output='screen',
        parameters=[{
            'yolo_conf_threshold':     0.25,
            'high_motion_threshold':   0.15,
            'medium_motion_threshold': 0.05,
            'low_motion_threshold':    0.02,
            'motion_normalizer':       255.0,
            'fusion_rate':             20.0,
        }]
    )

    tracking_node = Node(
        package='semi_active_suspension_system',
        executable='tracking_node',
        name='tracking_node',
        output='screen',
        parameters=[{
            'max_missing_frames': 5,
            'min_hits':           2,
            'iou_threshold':      0.2,
        }]
    )

    estimation_node = Node(
        package='semi_active_suspension_system',
        executable='estimation_node',
        name='estimation_node',
        output='screen',
        parameters=[{
            'focal_length_px':         focal_length_px,
            'assumed_pothole_width_m': 0.40,
            'depth_scale':             18.0,
            'min_distance_m':          0.5,
            'max_distance_m':          15.0,
            'track_timeout_s':         1.0,
        }]
    )

    controller_node = Node(
        package='semi_active_suspension_system',
        executable='controller_node',
        name='controller_node',
        output='screen',
        parameters=[{
            'mid_zone_left':         mid_zone_left,
            'mid_zone_right':        mid_zone_right,
            'image_width':           image_width,
            'image_height':          image_height,
            'serial_port':           serial_port,
            'baud_rate':             baud_rate,
            'min_conf_to_act':       0.50,
            'idle_pwm':              0,
            'preload_pwm':           80,
            'max_pwm':               200,
            'control_rate':          20.0,
            'actuator_lag_sec':      0.20,
            'impact_hold_sec':       0.40,
            'bottom_y_impact_frac':  0.88,
            'bottom_y_trigger_frac': 0.70,
        }]
    )

    visualization_node = Node(
        package='semi_active_suspension_system',
        executable='visualization_node',
        name='visualization_node',
        output='screen',
        parameters=[{
            'show_motion_score':   True,
            'show_track_id':       True,
            'show_confidence_bar': True,
            'show_estimation':     True,
            'show_mid_zone':       True,
            'show_controller_hud': True,
            'mid_zone_left':       mid_zone_left,
            'mid_zone_right':      mid_zone_right,
            'image_width':         image_width,
            'vis_rate':            20.0,
        }]
    )

    return LaunchDescription(
        args + [
            camera_node,
            preprocessing_node,
            yolo_node,
            optical_flow_node,
            TimerAction(period=3.0, actions=[
                fusion_node,
                tracking_node,
                estimation_node,
                controller_node,
                visualization_node,
            ]),
        ]
    )
import os
import cv2
import torch
import numpy as np
import supervision as sv
import open3d as o3d
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Float32MultiArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
import rclpy
from typing import Tuple
from builtin_interfaces.msg import Time
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CameraInfo
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R

"""
Hyper parameters
"""
GROUNDING_MODEL = "IDEA-Research/grounding-dino-base"
OBJECT = "small red cube ."
TARGET = "basket ."
SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSING_INTERVAL_SEC = 5.0  # Processing interval to throttle image processing

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')

        # Publishers and Subscribers
        self.publisher = self.create_publisher(Float32MultiArray, '/object_state', 1)
        
        # Set up a low-frequency subscription
        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1
        )

        self.color_subscription = self.create_subscription(
            RosImage, '/color_image5', self.color_callback, qos_profile)
        
        self.depth_subscription = self.create_subscription(
            RosImage, '/depth_image5', self.depth_callback, qos_profile)

        # Subscribe to the camera info topic
        self.camera_info_subscription = self.create_subscription(
            CameraInfo, '/color_camera_info5', self.camera_info_callback, qos_profile)

        # Image processing utilities
        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.camera_info = None
        self.last_processed_time = self.get_clock().now()

        # tf2 listener for extrinsic matrix
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # SAM2 and Grounding DINO setup
        self.sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        self.processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)

        # Flag to alternate between processing OBJECT and TARGET
        self.is_processing_object = True
        self.object_detected = False

    def camera_info_callback(self, msg: CameraInfo):
        """Callback function for camera info topic."""
        self.camera_info = msg

    def get_intrinsic_from_camera_info(self):
        """Extract camera intrinsic parameters from the CameraInfo message."""
        if self.camera_info is None:
            return None
        
        width = self.camera_info.width
        height = self.camera_info.height
        fx = self.camera_info.k[0]  # Focal length in x-axis
        fy = self.camera_info.k[4]  # Focal length in y-axis
        cx = self.camera_info.k[2]  # Principal point x
        cy = self.camera_info.k[5]  # Principal point y
        
        return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    def get_extrinsic_from_tf(self, timeout_sec=10.0, retry_interval_sec=0.5):
        """Get the extrinsic matrix from the TF transform between camera and robot with retry logic."""
        start_time = self.get_clock().now()

        while (self.get_clock().now() - start_time).nanoseconds * 1e-9 < timeout_sec:
            try:
                transform: TransformStamped = self.tf_buffer.lookup_transform(
                    'RGBDCamera5', 'panda_link0', rclpy.time.Time())

                translation = transform.transform.translation
                rotation = transform.transform.rotation

                quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
                rotation_matrix = R.from_quat(quaternion).as_matrix()

                extrinsic_matrix = np.eye(4)
                extrinsic_matrix[0:3, 0:3] = rotation_matrix
                extrinsic_matrix[0:3, 3] = [translation.x, translation.y, translation.z]

                return extrinsic_matrix

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.get_logger().warn('TF lookup failed, retrying...')

        self.get_logger().error(f"TF lookup failed after {timeout_sec} seconds.")
        return None

    def color_callback(self, msg):
        current_time = self.get_clock().now()
        if (current_time - self.last_processed_time).nanoseconds * 1e-9 < PROCESSING_INTERVAL_SEC:
            return  # Skip processing if within the throttling interval

        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgba8')
        if self.color_image is not None and self.depth_image is not None and self.camera_info is not None:
            self.process_images()
            self.last_processed_time = current_time

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def process_images(self):
        """Process the color and depth images for object detection and point cloud generation."""
        image_pil = Image.fromarray(self.color_image).convert("RGB")
        panda_intrinsic = self.get_intrinsic_from_camera_info()
        if panda_intrinsic is None:
            self.get_logger().warn("Camera intrinsic parameters not available.")
            return

        extrinsic = self.get_extrinsic_from_tf()

        torch.autocast(device_type=DEVICE, dtype=torch.float16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Determine if processing OBJECT or TARGET based on the flag
        if self.is_processing_object:
            text = OBJECT
        else:
            if not self.object_detected:
                self.get_logger().info("Skipping TARGET processing because no OBJECT was detected.")
                self.is_processing_object = not self.is_processing_object  # Switch back to OBJECT
                return
            text = TARGET

        with torch.no_grad():
            self.sam2_predictor.set_image(np.array(image_pil))

            inputs = self.processor(images=image_pil, text=text, return_tensors="pt").to(DEVICE)
            outputs = self.grounding_model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.3,
                text_threshold=0.3,
                target_sizes=[image_pil.size[::-1]]
            )

            input_boxes = results[0]["boxes"].cpu().numpy()

        if len(input_boxes) > 0:
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            if masks.ndim == 4:
                masks = masks.squeeze(1)

            detections = sv.Detections(
                xyxy=input_boxes,
                mask=masks.astype(bool),
                class_id=np.array(list(range(len(results[0]["labels"]))))
            )

            masked_depth_image = np.zeros_like(self.depth_image, dtype=np.float32)
            mask = detections.mask[0]
            masked_depth_image[mask] = self.depth_image[mask]

            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                o3d.geometry.Image(masked_depth_image), panda_intrinsic, extrinsic
            )

            # Remove outliers
            _, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=2.0)
            filtered_pcd = pcd.select_by_index(ind)

            points = np.asarray(filtered_pcd.points)

            if self.is_processing_object:
                # Calculate full 3D centroid for OBJECT
                centroid = np.mean(points, axis=0)
                centroid[2] += 0.1  # Added a small offset to prevent gripper collision
                self.get_logger().info(f'Calculated 3D centroid for OBJECT: {centroid}')
                self.object_detected = True
                gripper_state = 0.0
            else:
                # Calculate central point in XY plane and highest Z point for TARGET
                xy_center = np.mean(points[:, :2], axis=0)  # Mean of x and y coordinates
                max_z = np.max(points[:, 2])  # Maximum of z coordinates
                centroid = np.array([xy_center[0], xy_center[1], max_z + 0.1])
                self.get_logger().info(f'Calculated central XY and highest Z for TARGET: {centroid}')
                gripper_state = 1.0
            z = 0.0
            state = np.array([centroid[0], centroid[1], centroid[2], z, gripper_state])
            # Publish the calculated centroid
            self.publish_state(state)
            # o3d.visualization.draw_geometries([filtered_pcd])
        else:
            if self.is_processing_object:
                # If no OBJECT was found, set object_detected to False
                self.object_detected = False
            self.get_logger().info(f"No detections found for {text}.")

        # Toggle the flag to alternate between OBJECT and TARGET
        self.is_processing_object = not self.is_processing_object

    def publish_state(self, centroid):
        msg = Float32MultiArray()
        msg.data = list(centroid)
        self.publisher.publish(msg)
        self.get_logger().info(f'Published centroid: {centroid}')


def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
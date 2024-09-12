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
TEXT_PROMPT = "small red cube ."
SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/my_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSING_INTERVAL_SEC = 2.0  # Processing interval to throttle image processing

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

    def camera_info_callback(self, msg: CameraInfo):
        """Callback function for camera info topic."""
        self.camera_info = msg
        # self.get_logger().info(f"Camera Info received. Width: {msg.width}, Height: {msg.height}")

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
                # Try to lookup the transformation
                transform: TransformStamped = self.tf_buffer.lookup_transform(
                    'RGBDCamera5', 'panda_link0', rclpy.time.Time())

                # Extract rotation (quaternion) and translation from the transform
                translation = transform.transform.translation
                rotation = transform.transform.rotation

                # Convert quaternion to rotation matrix
                quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
                rotation_matrix = R.from_quat(quaternion).as_matrix()

                # Create the extrinsic matrix (4x4)
                extrinsic_matrix = np.eye(4)
                extrinsic_matrix[0:3, 0:3] = rotation_matrix
                extrinsic_matrix[0:3, 3] = [translation.x, translation.y, translation.z]

                return extrinsic_matrix

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.get_logger().warn('TF lookup failed, retrying...')
                # Wait for retry_interval_sec seconds before retrying
                rclpy.sleep(retry_interval_sec)

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
        # Grounding DINO inference
        image_pil = Image.fromarray(self.color_image)
        image_pil = image_pil.convert("RGB")

        # Get the intrinsic parameters from camera info
        panda_intrinsic = self.get_intrinsic_from_camera_info()
        if panda_intrinsic is None:
            self.get_logger().warn("Camera intrinsic parameters not available.")
            return

        # Get the extrinsic parameters from TF
        extrinsic = self.get_extrinsic_from_tf()

        # environment settings
        torch.autocast(device_type=DEVICE, dtype=torch.float16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # setup the input image and text prompt for SAM 2 and Grounding DINO
        text = TEXT_PROMPT

        image = image_pil
        with torch.no_grad():
            self.sam2_predictor.set_image(np.array(image.convert("RGB")))

            inputs = self.processor(images=image, text=text, return_tensors="pt").to(DEVICE)
        
            outputs = self.grounding_model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
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

            # Visualize and process image
            detections = sv.Detections(
                xyxy=input_boxes,
                mask=masks.astype(bool),
                class_id=np.array(list(range(len(results[0]["labels"]))))
            )

            masked_depth_image = np.zeros_like(self.depth_image, dtype=np.float32)
            for mask in detections.mask:
                masked_depth_image[mask] = self.depth_image[mask]

            # Generate the point cloud using the intrinsic and extrinsic parameters
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                o3d.geometry.Image(masked_depth_image), panda_intrinsic, extrinsic
            )

            # Remove outliers
            _, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=2.0)
            filtered_pcd = pcd.select_by_index(ind)

            # Calculate centroid
            points = np.asarray(filtered_pcd.points)
            centroid = np.mean(points, axis=0)
            centroid[2] += 0.1  # Added a small offset to the z-coordinate to prevent hitting the object by the gripper

            # Publish centroid
            self.publish_centroid(centroid)
            del inputs, outputs, results, masks, scores, logits, input_boxes
        else:
            self.get_logger().info("No detections found.")

    def publish_centroid(self, centroid):
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
#!/usr/bin/env python

###########
# IMPORTS #
###########
import numpy as np
import rclpy
from rclpy.node import Node
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from line_interfaces.msg import Line
import sys
import math

#############
# CONSTANTS #
#############
LOW = 200  # Lower image thresholding bound
HI = 255   # Upper image thresholding bound
LENGTH_THRESH = 0  # If the length of the largest contour is less than LENGTH_THRESH, we will not consider it a line
KERNEL = np.ones((5, 5), np.uint8)
DISPLAY = True

# Constants from your tracker
EXTEND = 400

class LineDetector(Node):
    def __init__(self):
        super().__init__('detector')

        # A subscriber to the topic '/aero_downward_camera/image'
        self.camera_sub = self.create_subscription(
            Image,
            '/world/line_following_track/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image',
            self.camera_sub_cb,
            10
        )

        # A publisher which will publish a parametrization of the detected line to the topic '/line/param'
        self.param_pub = self.create_publisher(Line, '/line/param', 1)

        # A publisher which will publish an image annotated with the detected line to the topic 'line/detector_image'
        self.detector_image_pub = self.create_publisher(Image, '/line/detector_image', 1)

        # Initialize instance of CvBridge to convert images between OpenCV images and ROS images
        self.bridge = CvBridge()
        
        # Store the latest image for visualization
        self.latest_image = None
        self.latest_line = None

    ######################
    # CALLBACK FUNCTIONS #
    ######################
    def camera_sub_cb(self, msg):
        """
        Callback function which is called when a new message of type Image is received by self.camera_sub.
            Args: 
                - msg = ROS Image message
        """
        # Convert Image msg to OpenCV image
        image = self.bridge.imgmsg_to_cv2(msg, "mono8")
        self.latest_image = image.copy()

        # Detect line in the image. detect returns a parameterize the line (if one exists)
        line = self.detect_line(image)
        self.latest_line = line

        # If a line was detected, publish the parameterization to the topic '/line/param'
        if line is not None:
            msg = Line()
            msg.x, msg.y, msg.vx, msg.vy = float(line[0]), float(line[1]), float(line[2]), float(line[3])
            # Publish param msg
            self.param_pub.publish(msg)

        # Publish annotated image with enhanced visualization
        if DISPLAY and line is not None:
            self.publish_annotated_image()

    def publish_annotated_image(self):
        """Create and publish an annotated image with line detection and tracking visualization"""
        if self.latest_image is None or self.latest_line is None:
            return
            
        # Convert to color image
        annotated = cv2.cvtColor(self.latest_image, cv2.COLOR_GRAY2BGR)
        
        # Get actual image dimensions
        image_height, image_width = self.latest_image.shape
        center_x = image_width // 2
        center_y = image_height // 2
        
        x, y, vx, vy = self.latest_line
        
        # Original line visualization (red line)
        pt1 = (int(x - 100*vx), int(y - 100*vy))
        pt2 = (int(x + 100*vx), int(y + 100*vy))
        cv2.line(annotated, pt1, pt2, (0, 0, 255), 2)  # Red line
        cv2.circle(annotated, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green center point
        
        # Add tracking visualization
        line_point = np.array([float(x), float(y)])
        line_dir = np.array([float(vx), float(vy)])
        line_dir = line_dir / np.linalg.norm(line_dir)  # Normalize to unit vector
        
        # Calculate target point (same logic as in tracker)
        target = line_point + EXTEND * line_dir
        
        # Draw line direction vector (blue arrow)
        arrow_end = line_point + 50 * line_dir  # Scale for visibility
        cv2.arrowedLine(annotated, 
                       (int(line_point[0]), int(line_point[1])), 
                       (int(arrow_end[0]), int(arrow_end[1])), 
                       (255, 0, 0), 3, tipLength=0.3)  # Blue arrow
        
        # Draw target point (yellow circle)
        target_x, target_y = float(target[0]), float(target[1])
        if 0 <= target_x < image_width and 0 <= target_y < image_height:
            cv2.circle(annotated, (int(target_x), int(target_y)), 8, (0, 255, 255), -1)  # Yellow circle
        
        # Draw center point (white circle) - THIS SHOULD BE FIXED AT IMAGE CENTER
        cv2.circle(annotated, (center_x, center_y), 6, (255, 255, 255), 2)  # White circle (outline)
        
        # Calculate error vector
        error_x = target_x - center_x
        error_y = target_y - center_y
        
        # Draw error vector from center to target (cyan line)
        cv2.line(annotated, 
                (center_x, center_y), 
                (int(target_x), int(target_y)), 
                (255, 255, 0), 2)  # Cyan line
        
        # Add text annotations
        cv2.putText(annotated, f'Image Size: {image_width}x{image_height}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f'Center: ({center_x}, {center_y})', 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f'Error: ({error_x:.1f}, {error_y:.1f})', 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Calculate angle information
        forward = np.array([0.0, 1.0])
        angle = math.atan2(float(line_dir[1]), float(line_dir[0]))
        angle_error = math.atan2(forward[1], forward[0]) - angle
        angle_error_deg = float(math.degrees(angle_error))
        cv2.putText(annotated, f'Angle Error: {angle_error_deg:.1f}°', 
                   (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add legend at bottom
        legend_y = image_height - 120
        cv2.putText(annotated, 'Legend:', (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 20
        cv2.putText(annotated, 'Red Line: Detected Line', (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        legend_y += 18
        cv2.putText(annotated, 'Blue Arrow: Line Direction', (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        legend_y += 18
        cv2.putText(annotated, 'Yellow Circle: Target Point', (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        legend_y += 18
        cv2.putText(annotated, 'White Circle: Image Center', (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 18
        cv2.putText(annotated, 'Cyan Arrow: Error Vector', (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Convert to ROS Image message and publish
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
        self.detector_image_pub.publish(annotated_msg)

    ##########
    # DETECT #
    ##########
    def detect_line(self, image):
        """ 
        Given an image, fit a line to biggest contour if it meets size requirements (otherwise return None)
        and return a parameterization of the line as a center point on the line and a vector
        pointing in the direction of the line.
            Args:
                - image = OpenCV image
            Returns: (x, y, vx, vy) where (x, y) is the centerpoint of the line in image and 
            (vx, vy) is a vector pointing in the direction of the line. Both values are given
            in downward camera pixel coordinates. Returns None if no line is found
        """

        '''
        TODO: Implement computer vision to detect a line (look back at last week's labs)
        TODO: Retrieve x, y pixel coordinates and vx, vy collinear vector from the detected line (look at cv2.fitLine)
        TODO: Populate the Line custom message and publish it to the topic '/line/param'
        '''
        #dilate
        imgray = image.copy()
        imgray = cv2.dilate(imgray,KERNEL,iterations = 1)

        #binary threshold and find contours
        ret, thresh = cv2.threshold(imgray, LOW, HI, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No contours found")
            return None

        max_contour = max(contours, key=lambda c: max(cv2.boundingRect(c)[2:4]))

        if max(cv2.boundingRect(max_contour)[2:4]) < LENGTH_THRESH:
            print("Largest contour too small")
            return None

        [vx,vy,x,y] = cv2.fitLine(max_contour, cv2.DIST_L2,0,0.01,0.01)

        return (x, y, vx, vy)


def main(args=None):
    rclpy.init(args=args)
    detector = LineDetector()
    detector.get_logger().info("Line detector initialized")
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        print("Shutting down")
    except Exception as e:
        print(e)
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
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
HEADING = 1.57

# Constants from your tracker
EXTEND = 350

def detect_white_strict(img):
   """Detect white pixels with strict thresholds"""
   lower_white = np.array([250, 250, 250])
   upper_white = np.array([255, 255, 255])
   mask = cv2.inRange(img, lower_white, upper_white)
   return mask, np.sum(mask > 0) > 100

def detect_white_relaxed(img):
   """Detect white pixels with relaxed thresholds"""
   lower_white = np.array([200, 200, 200])
   upper_white = np.array([255, 255, 255])
   mask = cv2.inRange(img, lower_white, upper_white)
   return mask, np.sum(mask > 0) > 100

def detect_brightest_pixels(img):
   """Detect the brightest pixels in the image"""
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   threshold = np.percentile(gray, 90)  # Top 10% brightest pixels
   _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
   return mask, np.sum(mask > 0) > 100

def detect_adaptive_threshold(img):
   """Use adaptive thresholding"""
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
   return mask, np.sum(mask > 0) > 100

def detect_color_edges(img):
   """Detect edges and assume they might be lines"""
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   edges = cv2.Canny(gray, 50, 150)
   # Dilate edges to make them thicker
   kernel = np.ones((3, 3), np.uint8)
   mask = cv2.dilate(edges, kernel, iterations=2)
   return mask, np.sum(mask > 0) > 100

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

        # self.camera_sub = self.create_subscription(
        #     Image,
        #     '/camera_0/image_raw',
        #     self.camera_sub_cb,
        #     10
        # )

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
        image_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
       
        self.latest_image = image_color.copy()

        # Detect line in the image. detect returns a parameterize the line (if one exists)
        line = self.detect_line(image_color)
        self.latest_line = line

        msg = Line()

        # If a line was detected, publish the parameterization to the topic '/line/param'
        if line is not None:
            msg.x, msg.y, msg.vx, msg.vy = float(line[0]), float(line[1]), float(line[2]), float(line[3])
            # Publish param msg
        else: 
            msg.x, msg.y, msg.vx, msg.vy = 0.0, 0.0, 0.0, 0.0
        self.param_pub.publish(msg) 


        # Publish annotated image with enhanced visualization
        if DISPLAY and line is not None:
            self.publish_annotated_image()

    def publish_annotated_image(self):
        """Create and publish an annotated image with line detection and tracking visualization"""
        gray = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2GRAY)
        if self.latest_image is None or self.latest_line is None:
            return
        
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Get actual image dimensions
        image_height, image_width, _ = self.latest_image.shape
        center_x = image_width // 2
        center_y = image_height // 2
        self.get_logger().info(f"CENTER {center_x}, {center_y}")
        
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
        centre 	= np.array([float(center_x), float(center_y)])
        c      	= centre - line_point
        t_along	= np.dot(c, line_dir)           	# scalar
        proj_pt	= line_point + t_along * line_dir   # (x,y) of projection


        # ----- 2. draw the perpendicular (cross-track) segment -------------
        cv2.line(annotated,
                (center_x, center_y),             	# image centre
                (int(proj_pt[0]), int(proj_pt[1])),   # projection
                (255,   0, 255), 2)               	# magenta line


        cv2.circle(annotated,                      	# mark projection point
                (int(proj_pt[0]), int(proj_pt[1])),
                4, (255,   0, 255), -1)






        
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
        
        
        # Try multiple detection strategies
        detection_methods = [
            ("white_strict", lambda: detect_white_strict(image)),
            ("white_relaxed", lambda: detect_white_relaxed(image)),
            ("brightest_pixels", lambda: detect_brightest_pixels(image)),
            ("adaptive_threshold", lambda: detect_adaptive_threshold(image)),
            ("color_edges", lambda: detect_color_edges(image))
        ]
        
        for method_name, method_func in detection_methods:
            try:
                binary, valid = method_func()
                white_pixels = np.sum(binary > 0)
                self.get_logger().info(f"{white_pixels}")
                if valid:
                    break
            except Exception as e:
                self.get_logger().info(f"{e}")
                continue
        else:
            self.get_logger().info("No valid line detected")
            return None

        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.get_logger().info("No contours found")
            return None
        
        # Find the largest contour
        max_contour = max(contours, key=cv2.contourArea)
        
        # Check if contour is large enough
        if cv2.contourArea(max_contour) < 0:  # Minimum area threshold
            self.get_logger().info("Largest contour too small")
            return None
        
        # Fit line to the largest contour
        [vx,vy,x,y] = cv2.fitLine(max_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Debug: Print original fitted line
        # self.get_logger().info(f"Original fitted line: vx={vx[0]:.3f}, vy={vy[0]:.3f}")
        
        # Ensure the line vector points in the direction closest to HEADING
        # Convert HEADING to direction vector (accounting for OpenCV's inverted y-axis)
        target_vx = np.cos(HEADING)
        target_vy = np.sin(HEADING) 
        
        # self.get_logger().info(f"Target direction: vx={target_vx:.3f}, vy={target_vy:.3f}")
        
        # Calculate dot product to determine if we need to flip the vector
        dot_product = vx[0] * target_vx + vy[0] * target_vy
        
        # If dot product is negative, flip the direction vector
        if dot_product < 0:
            vx = -vx
            vy = -vy
            self.get_logger().info("Flipped vector direction")
        else:
            self.get_logger().info("Kept original vector direction")
        
        # Normalize the vector to ensure consistent magnitude
        magnitude = np.sqrt(vx[0]**2 + vy[0]**2)
        if magnitude > 0:
            vx = vx / magnitude
            vy = vy / magnitude
        
        # print(f"Final line vector: vx={vx[0]:.3f}, vy={vy[0]:.3f}")
        
        
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
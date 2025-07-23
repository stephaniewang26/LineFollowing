#!/usr/bin/env python3

import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from line_interfaces.msg import Line
import tf_transformations as tft

#############
# CONSTANTS #
#############
_RATE = 10 # (Hz) rate for rospy.rate
_MAX_SPEED = 1.0 # (m/s)
_MAX_CLIMB_RATE = 1.0 # m/s
_MAX_ROTATION_RATE = 3.0 # rad/s 
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CENTER = np.array([IMAGE_WIDTH//2, IMAGE_HEIGHT//2]) # Center of the image frame. We will treat this as the center of mass of the drone
EXTEND = 100 # Number of pixels forward to extrapolate the line
KP_X = 0.1
KP_Y = 0.1
KP_Z_W = 1.0
DISPLAY = True

#########################
# COORDINATE TRANSFORMS #
#########################
class CoordTransforms():

    def __init__(self):
        """
        Variable Notation:
            - v__x: vector expressed in "x" frame
            - q_x_y: quaternion of "x" frame with relative to "y" frame
            - p_x_y__z: position of "x" frame relative to "y" frame expressed in "z" coordinates
            - v_x_y__z: velocity of "x" frame with relative to "y" frame expressed in "z" coordinates
            - R_x2y: rotation matrix that maps vector represented in frame "x" to representation in frame "y" (right-multiply column vec)
    
        Frame Subscripts:
            - m = marker frame (x-right, y-up, z-out when looking at marker)
            - dc = downward-facing camera (if expressed in the body frame)
            - fc = forward-facing camera
            - bu = body up frame (x-forward, y-left, z-up, similar to ENU)
            - bd = body down frame (x-forward, y-right, z-down, similar to NED)
            - lenu = local East-North-Up world frame ("local" implies that it may not be aligned with east and north, but z is up)
            - lned = local North-East-Down world frame ("local" implies that it may not be aligned with north and east, but z is down)
        Rotation matrix:
            R = np.array([[       3x3     0.0]
                          [    rotation   0.0]
                          [     matrix    0.0]
                          [0.0, 0.0, 0.0, 0.0]])
            
            [[ x']      [[       3x3     0.0]  [[ x ]
             [ y']  =    [    rotation   0.0]   [ y ]
             [ z']       [     matrix    0.0]   [ z ]
             [0.0]]      [0.0, 0.0, 0.0, 0.0]]  [0.0]]
        """
        
        # Reference frames
        self.COORDINATE_FRAMES = {'lenu','lned','bu','bd','dc','fc'}
    
        self.WORLD_FRAMES = {'lenu', 'lned'}
    
        self.BODY_FRAMES = {'bu', 'bd', 'dc', 'fc'}
    
        self.STATIC_TRANSFORMS = {'R_lenu2lenu',
                                  'R_lenu2lned',
    
                                  'R_lned2lenu',
                                  'R_lned2lned', 
          
                                  'R_bu2bu', 
                                  'R_bu2bd',
                                  'R_bu2dc',
                                  'R_bu2fc',
          
                                  'R_bd2bu',
                                  'R_bd2bd',
                                  'R_bd2dc',
                                  'R_bd2fc',
          
                                  'R_dc2bu',
                                  'R_dc2bd',
                                  'R_dc2dc',
                                  'R_dc2fc',
    
                                  'R_fc2bu',
                                  'R_fc2bd',
                                  'R_fc2dc',
                                  'R_fc2fc'
                                  }

        ######################
        # ROTATION MATRICIES #
        ######################
       
        # local ENU -> local NED | local NED -> local NED 
        self.R_lenu2lned = self.R_lned2lenu = np.array([[0.0, 1.0, 0.0, 0.0],
                                                        [1.0, 0.0, 0.0, 0.0],
                                                        [0.0, 0.0,-1.0, 0.0],
                                                        [0.0, 0.0, 0.0, 0.0]])
       
    
        # body up -> body down | body down -> body up | body up -> downward camera | downward camera -> body up 
        self.R_bu2bd = self.R_bd2bu = self.R_bu2dc = self.R_dc2bu = np.array([[1.0, 0.0, 0.0, 0.0],
                                                                              [0.0,-1.0, 0.0, 0.0],
                                                                              [0.0, 0.0,-1.0, 0.0],
                                                                              [0.0, 0.0, 0.0, 0.0]])
    
    
        # self -> self (identity matrix) | downward camera -> body down | body down -> downward camera
        self.R_lenu2lenu = self.R_lned2lned = self.R_bu2bu = self.R_bd2bd = self.R_dc2dc = self.R_fc2fc = self.R_bd2dc = np.array([[1.0, 0.0, 0.0, 0.0],
                                                                                                                                                  [0.0, 1.0, 0.0, 0.0],
                                                                                                                                                  [0.0, 0.0, 1.0, 0.0],
                                                                                                                                                  [0.0, 0.0, 0.0, 0.0]])
        
        self.R_dc2bd = np.array([
            [0.0, 1.0, 0.0, 0.0],  # bd.x = dc.y
            [1.0, 0.0, 0.0, 0.0],  # bd.y = dc.x
            [0.0, 0.0, 1.0, 0.0],  # bd.z = dc.z
            [0.0, 0.0, 0.0, 0.0]
        ])
    
        # body up -> forward camera 
        self.R_bu2fc = np.array([[0.0,-1.0, 0.0, 0.0],
                                 [0.0, 0.0,-1.0, 0.0],
                                 [1.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0]])
    
        # forward camera -> body up
        self.R_fc2bu = np.array([[ 0.0, 0.0, 1.0, 0.0],
                                 [-1.0, 0.0, 0.0, 0.0],
                                 [ 0.0,-1.0, 0.0, 0.0],
                                 [ 0.0, 0.0, 0.0, 0.0]])


        # body down -> forward camera | downward camera -> forward camera
        self.R_bd2fc = self.R_dc2fc = np.array([[0.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 1.0, 0.0],
                                                [1.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0]])
    
        # forward camera -> body down | forward camera -> downward camera
        self.R_fc2bd = self.R_fc2dc = np.array([[0.0, 0.0, 1.0, 0.0],
                                                [1.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0]])
    
    
    def static_transform(self, v__fin, fin, fout):
        """
        Given a vector expressed in frame fin, returns the same vector expressed in fout.
            
            Args:
                - v__fin: 3D vector, (x, y, z), represented in fin coordinates 
                - fin: string describing input coordinate frame 
                - fout: string describing output coordinate frame 
        
            Returns
                - v__fout: a vector, (x, y, z) represent in fout coordinates
        """
        # Check if fin is a valid coordinate frame
        if fin not in self.COORDINATE_FRAMES:
            raise AttributeError('{} is not a valid coordinate frame'.format(fin))

        # Check if fout is a valid coordinate frame
        if fout not in self.COORDINATE_FRAMES:
            raise AttributeError('{} is not a valid coordinate frame'.format(fout))
        
        # Check for a static transformation exists between the two frames
        R_str = 'R_{}2{}'.format(fin, fout)
        if R_str not in self.STATIC_TRANSFORMS:
            raise AttributeError('No static transform exists from {} to {}.'.format(fin, fout))
        
        # v4__'' are 4x1 np.array representation of the vector v__''
        # Create a 4x1 np.array representation of v__fin for matrix multiplication
        v4__fin = np.array([[v__fin[0]],
                            [v__fin[1]],
                            [v__fin[2]],
                            [     0.0]])

        # Get rotation matrix
        R_fin2fout = getattr(self, R_str)

        # Perform transformation from v__fin to v__fout
        v4__fout = np.dot(R_fin2fout, v4__fin)
        
        return (v4__fout[0,0], v4__fout[1,0], v4__fout[2,0])


    def get_v__lenu(self, v__fin, fin, q_bu_lenu):
        """
        Given a vector expressed in frame fin, returns the same vector expressed in the local ENU frame. q_bu_lene
        is the quaternion defining the rotation from the local ENU frame to the body up frame.
                
            Args:
                - v__fin: 3D vector, (x, y, z), represented in fin coordinates 
                - fin: string describing input coordinate frame 
                - q_bu_lenu: quaternion defining the rotation from local ENU frame to the body up frame. Quaternions ix+jy+kz+w are represented as (x, y, z, w)
            
            Returns:
                - v__lenu: 3D vector, (x, y, z), represented in local ENU world frame
        """
        # Check if fin is a valid coordinate frame
        if fin not in self.COORDINATE_FRAMES:
            raise AttributeError('{} is not a valid coordinate frame'.format(fin))

        # Transformations from one world frame to another can be down with a static transform
        if fin in self.WORLD_FRAMES:
            return self.static_transform(v__fin, fin, 'lenu')

        # Transformtion from body frame to world frame 
        elif fin in self.BODY_FRAMES:
            # Convert vector v__fin to the body up frame
            v__bu = self.static_transform(v__fin, fin, 'bu')

            # Create a 4x1 np.array representation of v__bu for matrix multiplication
            v4__bu = np.array([[v__bu[0]],
                                [v__bu[1]],
                                [v__bu[2]],
                                [     0.0]])

            # Create rotation matrix from the quaternion
            R_bu2lenu = tft.quaternion_matrix(q_bu_lenu)

            # Perform transformation from v__bu to v__lenu
            v4__lenu = np.dot(R_bu2lenu, v4__bu)
        
            return (v4__lenu[0,0], v4__lenu[1,0], v4__lenu[2,0])

#########################
# COORDINATE TRANSFORMS #
#########################
# Create CoordTransforms instance
coord_transforms = CoordTransforms()

class LineController(Node):
    def __init__(self) -> None:
        super().__init__('line_controller')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.line_sub = self.create_subscription(
            Line, '/line/param', self.line_sub_cb, 1)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.takeoff_height = -3.0

        # Linear setpoint velocities in downward camera frame
        self.vx__dc = 0.0
        self.vy__dc = 0.0
        self.vz__dc = 0.0

        # Yaw setpoint velocities in downward camera frame
        self.wz__dc = 0.0

        # Quaternion representing the rotation of the drone's body frame in the NED frame. initiallize to identity quaternion
        self.quat_bu_lenu = (0, 0, 0, 1)

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.1, self.timer_callback)

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = True
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint(self, vx: float, vy: float, wz: float) -> None:
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [None, None, self.takeoff_height]
        if self.offboard_setpoint_counter < 100:
            msg.velocity = [0.0, 0.0, 0.0]
        else:
            msg.velocity = [vx, vy, None]
        msg.acceleration = [None, None, None]
        msg.yawspeed = wz
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing velocity setpoints {[vx, vy, wz]}")

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def convert_velocity_setpoints(self):
        # Set linear velocity (convert command velocity from downward camera frame to bd frame)
        vx, vy, vz = coord_transforms.static_transform((self.vx__dc, self.vy__dc, self.vz__dc), 'dc', 'bd')

        # Set angular velocity (convert command angular velocity from downward camera to bd frame)
        _, _, wz = coord_transforms.static_transform((0.0, 0.0, self.wz__dc), 'dc', 'bd')

        # enforce safe velocity limits
        if _MAX_SPEED < 0.0 or _MAX_CLIMB_RATE < 0.0 or _MAX_ROTATION_RATE < 0.0:
            raise Exception("_MAX_SPEED,_MAX_CLIMB_RATE, and _MAX_ROTATION_RATE must be positive")
        vx = min(max(vx,-_MAX_SPEED), _MAX_SPEED)
        vy = min(max(vy,-_MAX_SPEED), _MAX_SPEED)
        # vz = min(max(vz,-_MAX_CLIMB_RATE), _MAX_CLIMB_RATE)
        wz = min(max(wz,-_MAX_ROTATION_RATE), _MAX_ROTATION_RATE)

        return (vx, vy, wz)
    
    def timer_callback(self) -> None:
        """Callback function for the timer."""

        self.publish_offboard_control_heartbeat_signal()
        
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        self.offboard_setpoint_counter += 1
    
    def line_sub_cb(self, param):
        """
        Callback function which is called when a new message of type Line is recieved by self.line_sub.
        Notes:
        - This is the function that maps a detected line into a velocity 
        command
            
            Args:
                - param: parameters that define the center and direction of detected line
        """
        print("Following line")
        # Extract line parameters
        x, y, vx, vy = param.x, param.y, param.vx, param.vy
        line_point = np.array([x, y])
        line_dir = np.array([vx, vy])
        line_dir = line_dir / np.linalg.norm(line_dir)  # Ensure unit vector
        if line_dir[1] > 0:
            line_dir = -line_dir  # Flip direction to point "up" in the image

        # Find closest point on the line to the image center
        center = CENTER
        to_center = center - line_point
        proj_length = np.dot(to_center, line_dir)
        closest = line_point + proj_length * line_dir

        # Target point EXTEND pixels ahead along the line direction
        target = closest + EXTEND * line_dir

        # Error between center and target
        error = target - center
        error_z = self.takeoff_height - self.vehicle_local_position.z

        # Set linear velocities (downward camera frame)
        self.vx__dc = KP_X * error[0]
        self.vy__dc = KP_Y * error[1]

        # Get angle between x-axis and line direction
        forward = np.array([1.0, 0.0])
        angle = math.atan2(line_dir[1], line_dir[0])
        angle_error = math.atan2(forward[1], forward[0]) - angle

        # Set angular velocity (yaw)
        self.wz__dc = KP_Z_W * angle_error
        self.publish_trajectory_setpoint(*self.convert_velocity_setpoints())


def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = LineController()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
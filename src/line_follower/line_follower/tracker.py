#!/usr/bin/env python3

import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from line_interfaces.msg import Line

#############
# CONSTANTS #
#############
_RATE = 10 # (Hz) rate for rospy.rate
_MAX_SPEED = 1.0 # (m/s)
_MAX_CLIMB_RATE = 1.0 # m/s
_MAX_ROTATION_RATE = 2.0 # rad/s 
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CENTER = np.array([IMAGE_WIDTH//2, IMAGE_HEIGHT//2]) # Center of the image frame. We will treat this as the center of mass of the drone
EXTEND = None # Number of pixels forward to extrapolate the line
KP_X = 5.0
KP_Y = 5.0
KP_Z = 5.0
KP_Z_W = 1.0
DISPLAY = True

class CoordTransforms():
    def __init__(self):
        self.COORDINATE_FRAMES = {'lenu','lned','bu','bd','dc','fc'}
        self.WORLD_FRAMES = {'lenu', 'lned'}
        self.BODY_FRAMES = {'bu', 'bd', 'dc', 'fc'}
        self.STATIC_TRANSFORMS = {'R_lenu2lenu','R_lenu2lned','R_lned2lenu','R_lned2lned',
                                  'R_bu2bu','R_bu2bd','R_bu2dc','R_bu2fc',
                                  'R_bd2bu','R_bd2bd','R_bd2dc','R_bd2fc',
                                  'R_dc2bu','R_dc2bd','R_dc2dc','R_dc2fc',
                                  'R_fc2bu','R_fc2bd','R_fc2dc','R_fc2fc'}

        # ... (rotation matrices as before) ...

    def quaternion_matrix(self, q):
        """Return homogeneous rotation matrix from quaternion."""
        x, y, z, w = q
        n = x*x + y*y + z*z + w*w
        if n < np.finfo(float).eps:
            return np.identity(4)
        q = np.array([x, y, z, w], dtype=np.float64) * np.sqrt(2.0 / n)
        q = np.outer(q, q)
        return np.array([
            [1.0-q[1,1]-q[2,2],     q[0,1]-q[2,3],     q[0,2]+q[1,3], 0.0],
            [    q[0,1]+q[2,3], 1.0-q[0,0]-q[2,2],     q[1,2]-q[0,3], 0.0],
            [    q[0,2]-q[1,3],     q[1,2]+q[0,3], 1.0-q[0,0]-q[1,1], 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

    def static_transform(self, v__fin, fin, fout):
        if fin not in self.COORDINATE_FRAMES:
            raise AttributeError('{} is not a valid coordinate frame'.format(fin))
        if fout not in self.COORDINATE_FRAMES:
            raise AttributeError('{} is not a valid coordinate frame'.format(fout))
        R_str = 'R_{}2{}'.format(fin, fout)
        if R_str not in self.STATIC_TRANSFORMS:
            raise AttributeError('No static transform exists from {} to {}.'.format(fin, fout))
        v4__fin = np.array([[v__fin[0]], [v__fin[1]], [v__fin[2]], [0.0]])
        R_fin2fout = getattr(self, R_str)
        v4__fout = np.dot(R_fin2fout, v4__fin)
        return (v4__fout[0,0], v4__fout[1,0], v4__fout[2,0])

    def get_v__lenu(self, v__fin, fin, q_bu_lenu):
        if fin not in self.COORDINATE_FRAMES:
            raise AttributeError('{} is not a valid coordinate frame'.format(fin))
        if fin in self.WORLD_FRAMES:
            return self.static_transform(v__fin, fin, 'lenu')
        elif fin in self.BODY_FRAMES:
            v__bu = self.static_transform(v__fin, fin, 'bu')
            v4__bu = np.array([[v__bu[0]], [v__bu[1]], [v__bu[2]], [0.0]])
            R_bu2lenu = self.quaternion_matrix(q_bu_lenu)
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
            Line, '/line/param', self.line_sub_cb, qos_profile)

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
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint(self, vx: float, vy: float, vz: float, wz: float) -> None:
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.velocity = [vx, vy, vz]
        msg.yawspeed = wz
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing velocity setpoints {[vx, vy, vz, wz]}")

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
        """
        Continually publishes Twist commands in the local lenu reference frame.
        """

        # Create velocity setpoint

        # Set linear velocity (convert command velocity from downward camera frame to lenu)
        vx, vy, vz = coord_transforms.get_v__lenu((self.vx__dc, self.vy__dc, self.vz__dc), 
                                                    'dc', self.quat_bu_lenu)

        # Set angular velocity (convert command angular velocity from downward camera to lenu)
        _, _, wz = coord_transforms.get_v__lenu((0.0, 0.0, self.wz__dc), 
                                                'dc', self.quat_bu_lenu)

        # enforce safe velocity limits
        if _MAX_SPEED < 0.0 or _MAX_CLIMB_RATE < 0.0 or _MAX_ROTATION_RATE < 0.0:
            raise Exception("_MAX_SPEED,_MAX_CLIMB_RATE, and _MAX_ROTATION_RATE must be positive")
        vx = min(max(vx,-_MAX_SPEED), _MAX_SPEED)
        vy = min(max(vy,-_MAX_SPEED), _MAX_SPEED)
        vz = min(max(vz,-_MAX_CLIMB_RATE), _MAX_CLIMB_RATE)
        wz = min(max(wz,-_MAX_ROTATION_RATE), _MAX_ROTATION_RATE)

        return (vx, vy, vz, wz)
    
    def timer_callback(self) -> None:
        """Callback function for the timer."""
        self.publish_offboard_control_heartbeat_signal()
        
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        if self.offboard_setpoint_counter >= 10:
            return

        if self.vehicle_local_position.z > self.takeoff_height:
            self.publish_trajectory_setpoint(0.0, 0.0, -1.0, 0.0)

        if self.offboard_setpoint_counter < 11:
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
        if self.offboard_setpoint_counter < 11:
            return

        # Extract line parameters
        x, y, vx, vy = param.x, param.y, param.vx, param.vy
        line_point = np.array([x, y])
        line_dir = np.array([vx, vy])
        line_dir = line_dir / np.linalg.norm(line_dir)  # Ensure unit vector

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
        self.vz__dc = KP_Z * error_z

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
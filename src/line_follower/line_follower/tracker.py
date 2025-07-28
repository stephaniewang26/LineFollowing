#!/usr/bin/env python3

import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, VehicleAttitude
from line_interfaces.msg import Line
import transformations as tft

#############
# CONSTANTS #
#############
_RATE = 10 # (Hz) rate for rospy.rate
_MAX_SPEED = 0.7 # (m/s)
_MAX_CLIMB_RATE = 1.0 # m/s
_MAX_ROTATION_RATE = 5.0 # rad/s
IMAGE_HEIGHT = 576
IMAGE_WIDTH = 768
CENTER = np.array([IMAGE_WIDTH//2, IMAGE_HEIGHT//2]) # Center of the image frame. We will treat this as the center of mass of the drone
EXTEND = 300 # Number of pixels forward to extrapolate the line
KP_X = 0.002
KP_Y = 0.002
KP_Z_W = 0.02

DISPLAY = True

ENABLE_HORIZONTAL_VELOCITY = True
MAX_CORRECTION_FACTOR = 10.0

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
       
        self.R_dc2bd = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        self.R_lenu2lenu = np.eye(4)
        self.R_lned2lned = np.eye(4)
        self.R_bu2bu = np.eye(4)
        self.R_bd2bd = np.eye(4)
        self.R_dc2dc = np.eye(4)
        self.R_fc2fc = np.eye(4)

        self.R_bd2dc = self.R_dc2bd.T

        self.R_lenu2lned = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        self.R_lned2lenu = self.R_lenu2lned.T

        self.R_bu2bd = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        self.R_bd2bu = self.R_bu2bd.T

        self.R_bu2dc = np.dot(self.R_bu2bd, self.R_bd2dc)
        self.R_bu2fc = np.eye(4)
        self.R_bd2fc = np.dot(self.R_bd2bu, self.R_bu2fc)
        self.R_dc2bu = self.R_bu2dc.T
        self.R_dc2fc = np.dot(self.R_dc2bd, self.R_bd2fc)
        self.R_fc2bu = self.R_bu2fc.T
        self.R_fc2bd = self.R_bd2fc.T
        self.R_fc2dc = self.R_dc2fc.T
    
    
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


class LineController(Node):
    def __init__(self) -> None:
        super().__init__('line_controller')

        # Create CoordTransforms instance
        self.coord_transforms = CoordTransforms()

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
        self.vehicle_attitude_subscriber = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude', self.vehicle_attitude_callback, qos_profile)
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

        self.q_bd_lned = np.array([1.0, 0.0, 0.0, 0.0])

        self.previous_line_dir = np.array([0.0, 1.0])
        self.direction_initialized = False

        self._last_vx = 0.0
        self._last_vy = 0.0
        self._last_wz = 0.0

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
        msg.attitude = False
        msg.body_rate = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint(self, vx: float, vy: float, wz: float) -> None:
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [None, None, self.takeoff_height]
        if self.offboard_setpoint_counter < 100:
            msg.velocity = [0.0, 0.0, 0.0]
        else:
            msg.velocity = [vx, vy, 0.0]
        msg.acceleration = [None, None, None]
        msg.yawspeed = wz
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        # self.get_logger().info(f"Publishing velocity setpoints {[vx, vy, wz]}")

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

    def vehicle_attitude_callback(self, msg):
        self.q_bd_lned = msg.q

    def get_current_heading(self):
        try:
            q = [
                self.q_bd_lned[1],
                self.q_bd_lned[2],
                self.q_bd_lned[3],
                self.q_bd_lned[0]
            ]

            euler = tft.euler_from_quaternion(q)
            yaw = euler[2]

            return yaw
        except (AttributeError, IndexError):
            return 0.0

    def convert_velocity_setpoints(self):
        vx_bd, vy_bd, _ = self.coord_transforms.static_transform((self.vx__dc, self.vy__dc, self.vz__dc), 'dc', 'bd')

        _, _, wz_bd = self.coord_transforms.static_transform((0.0, 0.0, self.wz__dc), 'dc', 'bd')

        yaw = self.get_current_heading()
        vx =  vx_bd * math.cos(yaw) - vy_bd * math.sin(yaw)
        vy =  vx_bd * math.sin(yaw) + vy_bd * math.cos(yaw)

        vx = min(max(vx,-_MAX_SPEED), _MAX_SPEED)
        vy = min(max(vy,-_MAX_SPEED), _MAX_SPEED)
        wz = min(max(wz_bd,-_MAX_ROTATION_RATE), _MAX_ROTATION_RATE)

        return (vx, vy, wz)

    def transform_and_publish_setpoints(self):
        v__bd = self.coord_transforms.static_transform(
            (self.vx__dc, self.vy__dc, self.vz__dc), 'dc', 'bd'
        )
        self.get_logger().info(f"DC frame commands: vx={self.vx__dc:.3f}, vy={self.vy__dc:.3f}, vz={self.vz__dc:.3f} | BD frame commands: vx={v__bd[0]:.3f}, vy={v__bd[1]:.3f}, vz={v__bd[2]:.3f}")

        q_px4 = self.q_bd_lned
        q_tf  = [q_px4[1], q_px4[2], q_px4[3], q_px4[0]]
        _, _, yaw = tft.euler_from_quaternion(q_tf)

        vx_ned =  v__bd[0]*np.cos(yaw) - v__bd[1]*np.sin(yaw)
        vy_ned =  v__bd[0]*np.sin(yaw) + v__bd[1]*np.cos(yaw)

        _, _, wz_bd = self.coord_transforms.static_transform(
            (0.0, 0.0, self.wz__dc), 'dc', 'bd'
        )
        self.get_logger().info(f"BD frame commands: vx={v__bd[0]:.3f}, vy={v__bd[1]:.3f}, vz={v__bd[2]:.3f} | NED frame commands: vx={vx_ned:.3f}, vy={vy_ned:.3f} | Yaw rate BD: wz={wz_bd:.3f}")

        velocity_magnitude = np.sqrt(vx_ned**2 + vy_ned**2)
        if velocity_magnitude > _MAX_SPEED:
            scale = _MAX_SPEED / velocity_magnitude
            vx_ned *= scale
            vy_ned *= scale

        wz_bd = min(max(wz_bd, -_MAX_ROTATION_RATE), _MAX_ROTATION_RATE)

        self._last_vx = vx_ned
        self._last_vy = vy_ned
        self._last_wz = wz_bd

    def timer_callback(self) -> None:
        """Callback function for the timer."""

        self.publish_offboard_control_heartbeat_signal()
        
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        self.offboard_setpoint_counter += 1
    
    def line_sub_cb(self, param):
        self.get_logger().info("Received line parameters")

        x, y, vx, vy = param.x, param.y, param.vx, param.vy

        norm = np.sqrt(vx**2 + vy**2)
        if norm > 0:
            vx /= norm
            vy /= norm

        target_x = x + EXTEND * vx
        target_y = y + EXTEND * vy

        error_x = target_x - CENTER[0]
        error_y = target_y - CENTER[1]

        self.vx__dc = KP_X * error_x / 100.0

        base_forward_speed = 0.5
        y_correction = -KP_Y * error_y / 200.0
        self.vy__dc = -(base_forward_speed + y_correction)

        desired_heading = np.arctan2(vx, -vy)
        angular_error = self.normalize_angle(desired_heading)
        self.wz__dc = KP_Z_W * angular_error / 10.0

        self.transform_and_publish_setpoints()

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


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
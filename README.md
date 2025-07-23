# Line Following Setup

## Requires
- PX4-Autopilot
  - https://github.com/px4/px4-autopilot/ (just clone for now, we will run the Gazebo sim from PX4-Autopilot directory)
- QGroundControl
  - https://docs.qgroundcontrol.com/master/en/qgc-user-guide/getting_started/download_and_install.html (follow installation)
- MicroXRCE Agent
  - https://docs.px4.io/main/en/middleware/uxrce_dds.html#micro-xrce-dds-agent-installation (follow installation)
- ros_gz_bridge 
```
sudo apt install ros-jazzy-ros-gz-bridge
```
- px4_msgs
  - included in this repo, just build it once
 


## Setting up the simulator

### In separate terminal sessions:

This will run the PX4 Gazebo sim with the line track
```
mv ~/LineFollowing/world/line_following_track.sdf ~/PX4-Autopilot/Tools/simulation/gz/worlds
cd PX4-Autopilot
PX4_GZ_WORLD=line_following_track make px4_sitl gz_x500_mono_cam_down
```
This will start up the agent that enables PX4 uORB and DDS comms

```
MicroXRCEAgent udp4 -p 8888
```
This will bridge the Gazebo camera stream to a ROS topic that we can use in our detector

```
ros2 run ros_gz_bridge parameter_bridge /world/line_following_track/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image@sensor_msgs/msg/Image@gz.msgs.Image 
```


Also start up the QGroundControl application

## Developing your line detector and flight controller

- Complete the TODOs in tracker.py and detector.py

When you run the simulator the first time, you will need to build the px4_msgs, line_interfaces, and line_follower packages
```
colcon build # in LineFollowing directory
```

When you are testing and tuning your detector/controller, you will only need to rebuild line_follower
```
colcon build --packages-select line_follower
```

You can run your detector and tracker nodes with the included launch file
```
ros2 launch line_follower follower_launch.py
```

Good luck!

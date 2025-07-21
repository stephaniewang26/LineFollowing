PX4_GZ_WORLD=line_following_track make px4_sitl gz_x500_mono_cam_down

MicroXRCEAgent udp4 -p 8888

ros2 run ros_gz_bridge parameter_bridge /world/line_following_track/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image@sensor_msgs/msg/Image@gz.msgs.Image 


#!/bin/bash

rosservice call /gazebo/reset_simulation "{}"
bash ../../judge/test_scripts/init_single_play.sh judge/marker_set/sim.csv localhost:5000 you enemy
rosnode kill /blue_bot/aruco_marker_publisher
rosnode kill /blue_bot/send_id_to_judge
rosnode kill /red_bot/aruco_marker_publisher
rosnode kill /red_bot/imageView
rosnode kill /red_bot/send_id_to_judge
# kill `ps auxww | grep '/usr/bin/python /opt/ros/kinetic/bin/roslaunch burger_war sim_robot_run.launch'| grep -v grep | awk '{ print $2}'| tail -1`
# sleep 20
roslaunch burger_war setup_sim_nospawn.launch &
bash ../../judge/test_scripts/set_running.sh localhost:5000
roslaunch burger_war sim_robot_run.launch &


# bash ../../judge/test_scripts/init_single_play.sh ../../marker_set/sim.csv localhost:5000 you enemy

# rostopic pub -1 /gazebo/set_model_state gazebo_msgs/ModelState '{model_name: red_bot, pose: { position: { x:  -0.00802941365317, y: -1.30180708272, z: -0.00100240389136 }, orientation: {x: -0.00289870472068, y: 0.00255000374612, z: 0.749537215393, w: 0.661950948132 } }, twist: { linear: { x: -2.14486099572e-05, y: -6.03070056406e-06, z: 3.57530023394e-05 }, angular: { x: 0.000172544678345, y: 5.45197249805e-05, z: 0.000337058115687}  }, reference_frame: world }'

# rostopic pub -1 /gazebo/set_model_state gazebo_msgs/ModelState '{model_name: blue_bot, pose: { position: { x:  0.00803057422089, y: 1.30182140949, z: -0.00100240392709 }, orientation: {x: 0.00254760493418, y: 0.00290081362037, z: -0.661330587132, w: 0.750084628234 } }, twist: { linear: { x: 2.14379168728e-05, y: 6.0664025918e-06, z: 3.57531181111e-05 }, angular: { x: -0.000172449078833, y: -5.47956803684e-05, z: 0.000337038447954}  }, reference_frame: world }'

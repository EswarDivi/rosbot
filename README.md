# RosBot 

## Self Driven Autonomous Differential Drive Robot 

Differential Drive Robot is a type of robot which has two wheels on either side of the robot. The wheels are driven by two motors which are connected to the wheels. The robot can move in any direction by controlling the speed of the motors. The robot can also turn by controlling the speed of the motors in opposite directions.

## Components

- Arduino Uno
- L298N Motor Driver
- 2 DC Motors
- 2 Wheels with 1 support wheel
- Nvidia Jetson Nano


## Architecture

Used Robotic Operating System (ROS) to control the robot. ROS is a set of software libraries and tools that help you build robot applications. From drivers to state-of-the-art algorithms, and with powerful developer tools, ROS has what you need for your next robotics project. And it's all open source.

With help of Deep Neural Network **Resnet18** Optimized with TensorRT, the robot can detect the objects in its path and avoid them. 
Used Opencv to collect the images from the camera and send it to the neural network. The neural network will return the whether there is object in the path or not. If there is an object in the path, the robot will stop and turn in the opposite direction.

## Demo

[![DEMO](https://img.youtube.com/vi/5uNJ9tkyM3k/0.jpg)](https://www.youtube.com/watch?v=5uNJ9tkyM3k)


## ROS Packages

- std_msgs
- rosserial_arduino
- rosserial_python
- roscpp
- rospy

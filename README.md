# Quadrotor-control-with-VIO

## Overview  
This project implements a quadrotor control system integrated with Visual-Inertial Odometry (VIO). It is intended for indoor testing and semi-autonomous flight scenarios. The main goals are:  
- to estimate the quadrotor’s pose (position & orientation) in real time using a VIO pipeline  
- to design and implement controller(s) (attitude & position) that use the VIO pose data  
- to demonstrate flight in simulation (and/or hardware) with the integrated system  

## Features  
- VIO pose estimation module (camera + IMU fusion)  
- State estimation output (pose + velocity) fed into controller  
- Attitude control loop (e.g., PID or other)  
- Position control logic for waypoint or trajectory following  
- Simulation support (e.g., via FlightSim folder)  
- Modular code structure for extension to outdoor or larger-scale flights

## Environment  
- Operating System: Ubuntu 20.04 LTS  
- Python: 3.8 – 3.10  
- ROS Distribution: Noetic Ninjemys  

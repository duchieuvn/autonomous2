# setup.py
from CONSTANTS import *

def setup_robot(robot):
    # Motors
    motors = {
        'fl': robot.getDevice("fl_wheel_joint"),
        'fr': robot.getDevice("fr_wheel_joint"),
        'rl': robot.getDevice("rl_wheel_joint"),
        'rr': robot.getDevice("rr_wheel_joint"),
    }

    for motor in motors.values():
        motor.setPosition(float('inf'))
        motor.setVelocity(0.0)

    # Position sensors
    sensors = {
        'fl': robot.getDevice("front left wheel motor sensor"),
        'fr': robot.getDevice("front right wheel motor sensor"),
        'rl': robot.getDevice("rear left wheel motor sensor"),
        'rr': robot.getDevice("rear right wheel motor sensor"),
    }

    for sensor in sensors.values():
        sensor.enable(TIME_STEP)

    # Cameras
    camera_rgb = robot.getDevice("camera rgb")
    camera_depth = robot.getDevice("camera depth")
    camera_rgb.enable(TIME_STEP)
    camera_depth.enable(TIME_STEP)

    # Lidar
    lidar = robot.getDevice("laser")
    lidar.enable(TIME_STEP)
    lidar.enablePointCloud()

    # IMU
    imu = {
        'accelerometer': robot.getDevice("imu accelerometer"),
        'gyro': robot.getDevice("imu gyro"),
        'compass': robot.getDevice("imu compass"),
    }
    for i in imu.values():
        i.enable(TIME_STEP)

    # Distance sensors
    distance_sensors = [
        robot.getDevice("fl_range"),
        robot.getDevice("rl_range"),
        robot.getDevice("fr_range"),
        robot.getDevice("rr_range")
    ]
    for sensor in distance_sensors:
        sensor.enable(TIME_STEP)

    return motors, sensors, imu, camera_rgb, camera_depth, lidar, distance_sensors

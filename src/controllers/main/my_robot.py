from controller import Supervisor
import numpy as np
from setup import setup_robot
import cv2
import random
import os
import sys
import pygame
import time
import utils
import threading
from CONSTANTS import *
from map import GridMap


class MyRobot(Supervisor):
    def __init__(self):
        super().__init__()
        self.motors, self.wheel_sensors, self.imu, self.camera_rgb, self.camera_depth, self.lidar, self.distance_sensors = setup_robot(self)
        self.time_step = TIME_STEP
        # Initialize GridMap instance
        self.map_object = GridMap(robot=self)
        # Backwards-compatibility: expose `grid_map` directly on robot for older helpers
        self.grid_map = self.map_object.grid_map
        self.wheel_radius = WHEEL_RADIUS
        self.axle_length = AXLE_LENGTH
        self.last_turn = 'right'
        self.start_point = None
        self.end_point = None
        self.blue_estimated_pos = None
        self.yellow_estimated_pos = None
        self.blue_pos_update_count = 0  # Track how many times blue column has been updated
        self.yellow_pos_update_count = 0  # Track how many times yellow column has been updated
        self.path = []
        self.interrupt_path = False
        self.chosen_frontier_count = 0
        # closure marking cooldown to avoid repeated marks when seeing same wall
        self.last_closure_time = 0.0
        self.closure_cooldown = 5.0  # seconds
        # Turning cooldown: track steps since turning stopped to allow sensor stabilization
        self.steps_since_turning = 0
        self.is_currently_turning = False
        # Multi-threading for continuous detection monitoring
        self.detection_thread = None
        self.camera_thread_running = False
        self.camera_detection_signal = None  # Shared variable for thread communication
        self.detection_lock = threading.Lock()  # Lock for thread synchronization
        # Multi-threading for lidar mapping
        self.lidar_thread = None
        self.lidar_thread_running = False
        self.lidar_lock = threading.Lock()  # Lock for thread synchronization
        # Multi-threading for stuck detection
        self.stuck_thread = None
        self.stuck_thread_running = False
        self.stuck_signal = False  # Flag raised by stuck thread
        self.stuck_last_position = None
        self.stuck_lock = threading.Lock()
        # Stuck detection for follow_local_target (other_branch)
        self.follow_target_last_position = None
        self.follow_target_stuck_count = 0
        self.follow_target_position_threshold = 0.005  # meters - minimum movement to not be considered stuck
        self.follow_target_stuck_threshold = 25  # number of calls without movement to consider stuck
        
        # --- Green carpet state (initialized to sensible defaults) ---
        # Camera intrinsics / extrinsics used by green-carpet projection
        try:
            self.cam_width = self.camera_rgb.getWidth()
            self.cam_height = self.camera_rgb.getHeight()
            self.cam_fov_rad = self.camera_rgb.getFov()
            self.fx = self.cam_width / (2.0 * np.tan(self.cam_fov_rad / 2.0))
            self.fy = self.fx
            self.cx = self.cam_width / 2.0
            self.cy = self.cam_height / 2.0
        except Exception:
            # Fallback safe defaults if camera not ready at init-time
            self.cam_width = 320
            self.cam_height = 240
            self.cam_fov_rad = 1.0472
            self.fx = 240.0
            self.fy = 240.0
            self.cx = self.cam_width / 2.0
            self.cy = self.cam_height / 2.0

        # Camera-to-robot offsets and height (tune as needed)
        self.camera_height_m = getattr(self, 'camera_height_m', 0.17)
        self.camera_pitch_rad = getattr(self, 'camera_pitch_rad', 0.0)
        self.X_offset = getattr(self, 'X_offset', 0.03)
        self.Y_offset = getattr(self, 'Y_offset', 0.0)

        # Persistent green-carpet tracking and cooldowns
        self.green_carpet_patches = []
        self.green_carpet_proximity_threshold = 200.0
        self.last_green_mark_time = 0.0
        self.green_mark_cooldown = 8.0
        self.last_green_carpet_points = []
        self.green_carpet_active = False

        self.last_found_color = None
        self.last_found_point = None

        # Track positions where column distance estimation was performed
        self.blue_estimation_positions = []  # List of positions where blue column was estimated
        self.yellow_estimation_positions = []  # List of positions where yellow column was estimated
        
        # Track previous estimation positions for distance checking (HIGHEST PRIORITY)
        self.blue_prev_estimate_position = None  # Previous position when blue was estimated
        self.yellow_prev_estimate_position = None  # Previous position when yellow was estimated
        self.estimation_distance_threshold = 0.5  # meters - minimum distance to perform another estimation

        # counters
        self.counter_obstacle_recoveries = 0

    def start_camera_thread(self):
        """Start the continuous detection thread for red walls and columns."""
        if self.detection_thread and self.detection_thread.is_alive():
            return
        self.camera_thread_running = True
        self.detection_thread = threading.Thread(target=self.camera_detection_loop, daemon=True)
        self.detection_thread.start()
        print("[Detection] Started continuous detection thread")

    def stop_camera_thread(self):
        """Stop the detection thread."""
        self.camera_thread_running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
            print("[Detection] Stopped detection thread")

    def start_lidar_thread(self):
        """Start the continuous lidar mapping thread."""
        if self.lidar_thread and self.lidar_thread.is_alive():
            return
        self.lidar_thread_running = True
        self.lidar_thread = threading.Thread(target=self.lidar_update_loop, daemon=True)
        self.lidar_thread.start()
        print("[Lidar] Started continuous lidar mapping thread")

    def stop_lidar_thread(self):
        """Stop the lidar mapping thread."""
        self.lidar_thread_running = False
        if self.lidar_thread:
            self.lidar_thread.join(timeout=1.0)
            print("[Lidar] Stopped lidar mapping thread")

    def start_stuck_thread(self):
        """Start the stuck detection thread."""
        if self.stuck_thread and self.stuck_thread.is_alive():
            return
        self.stuck_thread_running = True
        self.stuck_thread = threading.Thread(target=self.stuck_detection_loop, daemon=True)
        self.stuck_thread.start()
        print("[Stuck] Started stuck detection thread")

    def stop_stuck_thread(self):
        """Stop the stuck detection thread."""
        self.stuck_thread_running = False
        if self.stuck_thread:
            self.stuck_thread.join(timeout=1.0)
            print("[Stuck] Stopped stuck detection thread")
    
    def detect_green(self):
        """Detect green carpet in the bottom half of the camera image."""

        # Get sensor data
        hsv_img = self.get_bottom_half_hsv()
        
        if hsv_img is None:
            return None
        
        # Segment green carpet in image
        green_mask = utils.segment_color(hsv_img, 'green')
        
        # If enough green pixels are detected, return True
        green_pixels = cv2.countNonZero(green_mask)
        if green_pixels > 50:
            return True
        return False

    def camera_detection_loop(self):
        """Continuous detection loop with shared variable communication."""
        # Wait for sensors to initialize
        time.sleep(1.0)

        estimation_distance_threshold = 1.5
        
        while self.camera_thread_running:
            # Check if camera is ready
            if self.camera_rgb is None:
                time.sleep(0.5)
                continue
                
            # Only write to shared variable if it's None (main thread has processed previous signal)
            time.sleep(0.1)  # Check every 100ms
            with self.detection_lock:
                if self.camera_detection_signal is None:

                    # Check for red wall
                    red_wall = self.there_is_red_wall()
                    if red_wall:
                        self.camera_detection_signal = 'red_wall'
                        continue

                    green_detected = self.detect_green()
                    if green_detected:
                        self.camera_detection_signal = 'green_carpet'
                        continue
                        
                    # Check for columns
                    if self.found_all_2_columns():
                        continue
                    
                    # Check for columns
                    color = self.detect_column()
                    if color:
                        if color == 'blue' and self.start_point is not None:
                            continue
                        elif color == 'yellow' and self.end_point is not None:
                            continue

                        column_mask = utils.segment_color(self.get_hsv_image(), color)
                        column_distance = self.estimate_column_distance(color) 
                        
                        if self.column_close(column_mask):
                            self.center_column_in_view(color)
                            print(f"Column {color} is close", column_distance)
                            self.mark_column(color)
                            self.interrupt_path = True
                            self.turn_right_milisecond(600)
                        

                        else:
                            self.camera_detection_signal = ('column', color)

    def lidar_update_loop(self):
        """Continuous lidar mapping loop."""
        # Wait for sensors to initialize
        # time.sleep(1.0)
        
        while self.lidar_thread_running:
            try:
                # Check if lidar is ready
                if self.lidar is None:
                    time.sleep(0.5)
                    continue
                #(diff_other_branch)
                # DO NOT UPDATE MAP IF ROBOT IS NOT ON GROUND
                if not self.robot_on_ground():
                    time.sleep(0.1)
                    continue
                    
                # Only update map when robot is not turning
                if not self.is_turning():
                    with self.lidar_lock:
                        # self.stop_motor()
                        # self.step(self.time_step)
                        self.lidar_update_map()
                time.sleep(0.1)  # Update at same frequency as before

            except Exception as e:
                print(f"[Lidar] Error in lidar update loop: {e}")
                time.sleep(1.0)

    def stuck_detection_loop(self):
        """Continuous stuck detection loop."""
        # Wait for sensors to initialize
        time.sleep(1.0)
        last_position = self.get_position()
        while self.stuck_thread_running:
            with self.stuck_lock:  
                time.sleep(2.5)
                # Check if motors are running (any motor has non-zero velocity)
                motor_velocities = [motor.getVelocity() for motor in self.motors.values()]
                is_moving = any(abs(v) > 0.01 for v in motor_velocities)  # Small threshold to account for floating point
                
                if is_moving:
                    # Only check for stuck condition when motors are running
                    if self.robot_stuck(last_position, stuck_distance=0.07):
                        self.stuck_signal = True
                    else:
                        self.stuck_signal = False
                    last_position = self.get_position()
                else:
                    # Motors not running, don't flag as stuck and reset reference position
                    self.stuck_signal = False
                    last_position = self.get_position()                              
    
    def get_column_center_ratio(self, color):
        hsv_img = self.get_hsv_image()
        if hsv_img is None:
            return 0.0
        
        center_frame = hsv_img[:, hsv_img.shape[1]//3: 2*hsv_img.shape[1]//3]

        column_mask = utils.segment_color(center_frame, color)
        # nonzero_pixels = np.count_nonzero(column_mask)
        # ratio = nonzero_pixels / (column_mask.shape[0] * column_mask.shape[1])
        
        # ratio = length of column in center frame / height of center frame
        column_pixels_per_row = np.count_nonzero(column_mask, axis=1) # array of number of non-zero pixels per row [0,0,5,10,0,0,...]
        # column height = number of rows with non-zero pixels
        column_height = np.count_nonzero(column_pixels_per_row)
        ratio = column_height / column_mask.shape[0]        
        return ratio

    def get_column_center_pixels(self, color):
        hsv_img = self.get_hsv_image()
        if hsv_img is None:
            return 0.0
        
        center_frame = hsv_img[:, hsv_img.shape[1]//3: 2*hsv_img.shape[1]//3]

        column_mask = utils.segment_color(center_frame, color)
        # nonzero_pixels = np.count_nonzero(column_mask)
        # ratio = nonzero_pixels / (column_mask.shape[0] * column_mask.shape[1])
        
        # ratio = length of column in center frame / height of center frame
        column_pixels_per_row = np.count_nonzero(column_mask, axis=1) # array of number of non-zero pixels per row [0,0,5,10,0,0,...]
        # column height = number of rows with non-zero pixels
        column_height = np.count_nonzero(column_pixels_per_row)
        return column_height
        
    def column_close(self, column_mask):
        nonzero_pixels = np.count_nonzero(column_mask)
        ratio = nonzero_pixels / (column_mask.shape[0] * column_mask.shape[1])

        if ratio > 0.25:
            return True
        else:
            return False

    def is_turning(self):
        left_speed = self.motors['fl'].getVelocity()
        right_speed = self.motors['fr'].getVelocity()
        turning = abs(left_speed - right_speed) > 0.02
        return turning


    def stop_motor(self):
        for motor in self.motors.values():
            motor.setVelocity(0)

    def set_robot_velocity(self, left_speed, right_speed):
        self.motors['fl'].setVelocity(left_speed)
        self.motors['rl'].setVelocity(left_speed)
        self.motors['fr'].setVelocity(right_speed)
        self.motors['rr'].setVelocity(right_speed)

        if left_speed < right_speed:
            self.last_turn = 'left'
        if right_speed > left_speed:
            self.last_turn = 'right'
        

    def velocity_to_wheel_speeds(self, v, w):
        v_left = v - (self.axle_length / 2.0) * w
        v_right = v + (self.axle_length / 2.0) * w
        left_speed = v_left / self.wheel_radius
        right_speed = v_right / self.wheel_radius
        return left_speed, right_speed

    def get_heading(self, type='deg'):
        # Calculate the angle from robot direction to the x-axis
        orientation = self.getSelf().getOrientation()
        dir_x = orientation[0]
        dir_y = orientation[3]
        angle_rad = np.arctan2(dir_y, dir_x)
        if type == 'rad':
            return angle_rad
        elif type == 'deg':
            return np.degrees(angle_rad)

    def position_ahead(self, distance_m):
        """Calculate the position of a point ahead of the robot at given distance in meters."""

        current_x, current_y = self.get_position()
        heading_rad = self.get_heading('rad')
        
        # Calculate position ahead using heading angle
        ahead_x = current_x + distance_m * np.cos(heading_rad)
        ahead_y = current_y + distance_m * np.sin(heading_rad)
        
        return np.array([ahead_x, ahead_y])

    def get_angle_diff(self, map_target, type='deg'):
        heading = self.get_heading('rad')
        map_x, map_y = self.get_map_position()
        dx = map_target[0] - map_x
        dy = map_target[1] - map_y
        target_angle = np.arctan2(dy, dx)
        angle_diff = abs(target_angle - heading)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        return angle_diff if type == 'rad' else np.degrees(angle_diff)

    def get_distances(self):
        return [sensor.getValue() for sensor in self.distance_sensors]

    def get_position(self):
        return np.array(self.getSelf().getPosition()[:2])

    def get_map_position(self):
        x, y = self.get_position()
        map_x = self.map_object.map_size // 2 + int(x / RESOLUTION)
        map_y = self.map_object.map_size // 2 - int(np.ceil(y / RESOLUTION))
        return np.array([map_x, map_y])

    def get_map_distance(self, map_target):
        a = self.get_map_position()
        return np.linalg.norm(a - np.array(map_target))

    def convert_to_map_coordinates(self, x, y):
        map_x = self.map_object.map_size // 2 + int(x / RESOLUTION)
        map_y = self.map_object.map_size // 2 - int(np.ceil(y / RESOLUTION))
        return int(map_x), int(map_y)

    def convert_to_world_coordinates(self, map_x, map_y):
        x = (map_x - self.map_object.map_size // 2) * RESOLUTION
        y = (self.map_object.map_size // 2 - map_y) * RESOLUTION
        return float(x), float(y)
    
    # def obstacle_in_front(self):
    #     distances = self.get_distances()
    #     if min(distances[0], distances[2]) < 0.3:
    #         return True
    #     return False
    # Line 424 in your file
    def obstacle_in_front(self):
        # 1. Check original distance sensors (V-shape)
        ds_distances = self.get_distances()
        ds_obstacle = min(ds_distances[0], ds_distances[2]) < 0.05
        if ds_obstacle:
            print(f"[Virtual Bumper] Distance sensors detected obstacle at {min(ds_distances[0], ds_distances[2]):.2f}m")
        
        # 2. Check Lidar Virtual Bumper (The "Semicircle")
        lidar_dist = self.get_lidar_front_min_dist(angle_range_deg=35)
        lidar_obstacle = lidar_dist < 0.10  # Slightly larger threshold for safety
        if lidar_obstacle:
            print(f"[Virtual Bumper] Lidar detected obstacle at {lidar_dist:.2f}m")
        
        # Return True if either detects an obstacle
        return ds_obstacle or lidar_obstacle



    def turn_right_milisecond(self, s=200):  
        self.set_robot_velocity(8, -8)
        self.step(s)
        self.stop_motor()
        

    def turn_left_milisecond(self, s=200):
        self.set_robot_velocity(-8, 8)
        self.step(s)
        self.stop_motor()
        

    def move_backward_milisecond(self, s=300):
        self.set_robot_velocity(-7, -7)
        self.step(s)
        self.stop_motor()
        

    def adapt_direction(self):
        count = 0
        distances = self.get_distances()
        # (diff_other_branch)
        if np.mean(distances) < 0.3: # other branch is 0.16
            
            self.move_backward_milisecond(50)
            return

        while (min(distances[0], distances[2]) < OBSTACLE_AVOID_THRESHOLD and count < OBSTACLE_AVOID_MAX_ATTEMPTS):
            second = random.randint(TURN_DURATION_MIN, TURN_DURATION_MAX)
            if distances[0] < distances[2]:
                self.turn_right_milisecond(second)
            else:
                self.turn_left_milisecond(second)

            count += 1
            distances = self.get_distances()

    def dwa_planner(self, world_target):
        MAX_SPEED = MAX_VELOCITY * self.wheel_radius
        v_samples = DWA_VELOCITY_SAMPLES
        w_samples = DWA_ANGULAR_SAMPLES

        best_score = -float('inf')
        best_v = 0.0
        best_w = 0.0

        x, y = self.get_position()
        theta = self.get_heading('rad')
        current_distance = np.linalg.norm(world_target - np.array([x, y]))

        # Predict the robot's future position after 5 TIME_STEP
        dt = TIME_STEP / 1000 # Convert to TIME_STEP to seconds
        for v in v_samples:
            for w in w_samples:
                cx, cy, ct = x, y, theta
                good_path = True
                # Calculate the robot position (cx, cy) after 5 steps
                for _ in range(0, 2):
                    cx += v * np.cos(ct) * dt
                    cy += v * np.sin(ct) * dt
                    ct += w * dt

                    predicted_map_x, predicted_map_y = self.convert_to_map_coordinates(cx, cy)
                    if self.there_is_obstacle([predicted_map_x, predicted_map_y]):
                        good_path = False
                        break
                    
                    # If the furtue distance from the target is greater than the current distance
                    predicted_distance = np.linalg.norm(world_target - np.array([cx, cy]))
                    if predicted_distance - current_distance > DWA_PREDICTION_DISTANCE_THRESHOLD:
                        good_path = False
                        break

                if not good_path:
                    continue

                predicted_angle_to_target = np.arctan2(world_target[1]-cy, world_target[0]-cx)
                heading_error = utils.get_angle_diff(predicted_angle_to_target, ct)

                heading_score = np.cos(heading_error)
                distance_score = 1 - (predicted_distance / 2)
                speed_score = v / MAX_SPEED
                score = DWA_HEADING_WEIGHT * heading_score + DWA_DISTANCE_WEIGHT * distance_score + DWA_SPEED_WEIGHT * speed_score

                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w

        return best_v, best_w
    
    # def follow_local_target(self, map_target):
    #     # Return True if the robot reached the target
    #     # Return (reached_target, is_stuck)
    #     # Check if robot reached the target
    #     if self.get_map_distance(map_target) < PATH_FOLLOWING_TARGET_REACH_DISTANCE:
    #         # self.stop_motor()
    #         return True
    #         # Reset stuck detection on successful reach
    #         self.follow_target_last_position = None
    #         self.follow_target_stuck_count = 0
    #         return True, False

    #     # Check for stuck condition - robot not moving despite repeated calls
    #     current_position = self.get_position()
    #     if self.follow_target_last_position is not None:
    #         distance_moved = np.linalg.norm(current_position - self.follow_target_last_position)
    #         if distance_moved < self.follow_target_position_threshold:
    #             self.follow_target_stuck_count += 1
    #             if self.follow_target_stuck_count >= self.follow_target_stuck_threshold:
    #                 # Reset for next target
    #                 self.follow_target_stuck_count = 0
    #                 self.follow_target_last_position = None
    #                 return False, True  # Not reached, but stuck
    #         else:
    #             # Robot is moving, reset counter
    #             self.follow_target_stuck_count = 0

    #     self.follow_target_last_position = current_position
    #     return False, False

    def follow_local_target(self, map_target):
        # Return (reached_target, is_stuck)
        # Check if robot reached the target
        if self.get_map_distance(map_target) < PATH_FOLLOWING_TARGET_REACH_DISTANCE:
            # Reset stuck detection on successful reach
            self.follow_target_last_position = None
            self.follow_target_stuck_count = 0
            return True, False
        
        # Check for stuck condition - robot not moving despite repeated calls
        current_position = self.get_position()
        if self.follow_target_last_position is not None:
            distance_moved = np.linalg.norm(current_position - self.follow_target_last_position)
            if distance_moved < self.follow_target_position_threshold:
                self.follow_target_stuck_count += 1
                if self.follow_target_stuck_count >= self.follow_target_stuck_threshold:
                    # Reset for next target
                    self.follow_target_stuck_count = 0
                    self.follow_target_last_position = None
                    return False, True  # Not reached, but stuck
            else:
                # Robot is moving, reset counter
                self.follow_target_stuck_count = 0
        
        self.follow_target_last_position = current_position
        
        world_target = self.convert_to_world_coordinates(map_target[0], map_target[1])
        v, w = self.dwa_planner(world_target)
        left_speed, right_speed = self.velocity_to_wheel_speeds(v, w)
        self.set_robot_velocity(left_speed, right_speed)
        return False, False
    
    def get_lidar_front_min_dist(self, angle_range_deg=30):
        """Returns the minimum distance to an obstacle in a front-facing arc."""
        if self.lidar is None:
            return float('inf')
        
        # Get local 2D points (x is forward, y is left)
        points_local = self.get_pointcloud_2d()
        if len(points_local) == 0:
            return float('inf')
        
        # Calculate angles for all points
        # atan2(y, x) gives angle from x-axis (forward)
        angles = np.arctan2(points_local[:, 1], points_local[:, 0])
        distances = np.linalg.norm(points_local, axis=1)
        
        # Filter for points within the front arc (e.g., -30 to +30 degrees)
        limit = np.radians(angle_range_deg)
        front_mask = (angles > -limit) & (angles < limit)
        
        front_distances = distances[front_mask]
        if len(front_distances) == 0:
            return float('inf')
        
        return np.min(front_distances)

    def get_min_front_distance(self, angle_deg=None):
        if angle_deg is None:
            angle_deg = LIDAR_FRONT_CONE_ANGLE
        points = self.get_pointcloud_2d()

        # If the lidar pointcloud is empty, fall back to distance sensors
        if points.shape[0] == 0:
            sensor_vals = self.get_distances()
            if len(sensor_vals) == 0:
                return float('inf')
            # use the minimum of available front-facing sensors as a conservative estimate
            try:
                return float(min(sensor_vals[0], sensor_vals[2]))
            except Exception:
                return float(min(sensor_vals))

        angles = np.arctan2(points[:, 1], points[:, 0]) * 180.0 / np.pi
        mask = (angles > -angle_deg) & (angles < angle_deg)
        front_points = points[mask]

        # If no lidar points lie within the front cone, fall back to distance sensors
        if front_points.shape[0] == 0:
            sensor_vals = self.get_distances()
            if len(sensor_vals) == 0:
                return float('inf')
            try:
                return float(min(sensor_vals[0], sensor_vals[2]))
            except Exception:
                return float(min(sensor_vals))

        distances = np.linalg.norm(front_points, axis=1)
        return float(np.min(distances))

    def get_hsv_image(self):
        if self.camera_rgb is None:
            return None
            
        image_data = self.camera_rgb.getImage()
        if image_data is None:
            return None

        width = self.camera_rgb.getWidth()
        height = self.camera_rgb.getHeight()
        image = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
        frame = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return hsv
    
    def get_bottom_half_hsv(self):
        hsv = self.get_hsv_image()
        if hsv is None:
            return None

        height, width, _ = hsv.shape
        bottom_half = hsv[height // 2:, :]
        return bottom_half
    
    def get_camera_depth_cm(self):
        if self.camera_depth is None:
            return None
            
        depth_data = self.camera_depth.getRangeImage()
        if depth_data == None:
            return None
        
        w_d = self.camera_depth.getWidth()
        h_d = self.camera_depth.getHeight()
        depth_image = np.array(depth_data).reshape((h_d, w_d))
        depth_cm = depth_image * 100 # Meter -> Centimeter
        # Replace invalid infinite depths with -1
        depth_img = np.where(np.isinf(depth_cm), -1, depth_cm)
        # Use signed dtype to allow -1 sentinel
        depth_img = depth_img.astype(np.int16)
        return depth_img
    
    def there_is_red_wall(self):
        hsv = self.get_hsv_image()
        if hsv is None:
            return False

        height, width, _ = hsv.shape

        lower_red1 = np.array(RED_WALL_HSV_LOWER1)
        upper_red1 = np.array(RED_WALL_HSV_UPPER1)
        lower_red2 = np.array(RED_WALL_HSV_LOWER2)
        upper_red2 = np.array(RED_WALL_HSV_UPPER2)

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        red_pixels = cv2.countNonZero(red_mask)
        if red_pixels > width*height*COLOR_DETECTION_RED_PIXEL_RATIO:
            return True
        else:
            return False
        
    def turn_degrees(self, degree_to_turn, direction='right'):
        rad_to_turn = np.deg2rad(degree_to_turn)
        initial_heading = self.get_heading('rad')
        if direction == 'right':
            self.set_robot_velocity(8, -8)  # Start turning in place (right turn)
        else:
            self.set_robot_velocity(-8,8)

        while self.step(self.time_step) != -1:
            current_heading = self.get_heading('rad')
            angle_diff = abs(utils.get_angle_diff(current_heading, initial_heading))

            if angle_diff >= rad_to_turn - TURN_ANGLE_COMPLETION_THRESHOLD:
                break

        self.stop_motor()
        

    def mark_on_map(self, distance_cm, color='blue'):
        # Use provided distance if available; otherwise attempt to estimate.
        if distance_cm is None:
            distance_cm = self.estimate_column_distance(color)

        # If we still couldn't estimate the distance, skip marking.
        if distance_cm is None:
            return

        if distance_cm < COLOR_DETECTION_DEPTH_THRESHOLD:
            if color == 'blue':
                self.start_point = tuple(self.get_map_position()) # Blue
            elif color == 'yellow':
                self.end_point = tuple(self.get_map_position())  # Yellow

    def align_to_red_wall(self):

        # PD controller constants
        Kp = ALIGN_RED_WALL_KP
        Kd = ALIGN_RED_WALL_KD
        
        error_threshold = ALIGN_RED_WALL_ERROR_THRESHOLD
        last_error = 0
        backward_speed = RED_WALL_ALIGNMENT_SPEED
        align_step_count = 0
        while self.step(self.time_step) != -1:
            align_step_count += 1
            if align_step_count > 100:
                self.stop_motor()
                break
            hsv_img = self.get_hsv_image()
            if hsv_img is None:
                self.stop_motor()
                break

            height, width, _ = hsv_img.shape
            is_middle, red_mask = utils.red_is_middle(hsv_img)

            M = cv2.moments(red_mask)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                error = cX - (width // 2)

                # Check for completion
                if (abs(error) < error_threshold and is_middle):
                    self.stop_motor()
                    break

                # PD control for turning
                turn_speed = Kp * error + Kd * (error - last_error)
                last_error = error
                self.set_robot_velocity(backward_speed + turn_speed, backward_speed - turn_speed)
                # self.step(self.time_step)
            else: # No red detected
                self.stop_motor()
                break

    def mark_closure_block(self) -> bool:
        now = time.time()
        if now - self.last_closure_time < self.closure_cooldown:
            return False  # cooldown active

        try:
            forward_m = CLOSURE_MARK_FORWARD
            back_m = CLOSURE_MARK_BACKWARD
            width_m = CLOSURE_MARK_WIDTH

            ok = self.map_object.mark_closure_rect_simple(
                forward_m=forward_m,
                back_m=back_m,
                width_m=width_m
            )

            if ok:
                self.last_closure_time = now
            return bool(ok)

        except Exception as e:
            print(f"[warning] Failed to mark corridor closure: {e}")
            return False

    def is_column_fully_in_frame(self, hsv, color, top_strip_height=50):
        """
        Checks if the specified color (column) is fully inside the camera frame.
        Returns True if there are no color pixels in the top strip of the frame,
        indicating the column is fully visible (not cut off at the top).
        
        Parameters:
            hsv: HSV image from camera
            color: 'blue' or 'yellow'
            top_strip_height: Height of the top strip to check (in pixels)
        
        Returns:
            bool: True if no color pixels in top strip, False otherwise
        """
        if hsv is None:
            return False
        
        # Segment the color
        color_mask = utils.segment_color(hsv, color)
        
        # Check the top strip for any color pixels
        top_strip = color_mask[:top_strip_height, :]
        color_pixels_in_top = cv2.countNonZero(top_strip)
        
        # If no color pixels in top, column is fully in frame
        return color_pixels_in_top == 0

    def estimate_column_distance(self, color):
        """Estimate horizontal distance to detected column using depth and known height."""
        self.stop_motor()
        
        # Get sensor data
        depth_img = self.get_camera_depth_cm()
        hsv_img = self.get_hsv_image()
        
        if depth_img is None or hsv_img is None:
            return None
        
        # Segment column in image
        column_mask = utils.segment_color(hsv_img, color)
        if not np.any(column_mask):
            print(f"[Error] No {color} column detected in image.")
            return None
        
        # Extract valid depth values (filter out -1 sentinel)
        depth_values = depth_img[column_mask != 0]
        valid_depths = depth_values[depth_values > 0]
        
        if len(valid_depths) == 0:
            # We often get invalid depth when very close.
            # If the column occupies a lot of pixels, treat it as "close enough" and allow marking.
            area_ratio = float(np.count_nonzero(column_mask)) / float(column_mask.size)
            if area_ratio > 0.20:  # close in image -> likely right in front
                return 0.5  # force "close" behavior in caller
            return None # too far to estimate
        
        # Calculate horizontal distance using Pythagorean theorem
        max_depth_cm = float(np.max(valid_depths))
        if not self.is_column_fully_in_frame(hsv_img, color):
            print(f"Max depth: {max_depth_cm}")
            if max_depth_cm < 110.0:
                max_depth_cm = max_depth_cm * 1.25
            else:
                max_depth_cm = max_depth_cm * 1.1
        column_height_cm = 125.0
        
        if max_depth_cm <= column_height_cm:
            return np.mean(valid_depths)  # Fallback to average depth if max is too small
        
        horizontal_distance_cm = np.sqrt(max_depth_cm ** 2 - column_height_cm ** 2)
        return horizontal_distance_cm + 10.0

    def align_to_column(self, color):

        # PD controller constants
        Kp = ALIGN_COLUMN_KP
        Kd = ALIGN_COLUMN_KD
        
        error_threshold = ALIGN_COLUMN_ERROR_THRESHOLD
        last_error = 0
        forward_speed = ALIGN_COLUMN_FORWARD_SPEED

        while self.step(self.time_step) != -1:
            hsv_img = self.get_hsv_image()
            if hsv_img is None:
                self.stop_motor()
                break

            height, width, _ = hsv_img.shape
            area = height*width
            column_mask = utils.segment_color(hsv_img, color)
            M = cv2.moments(column_mask)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                error = cX - (width // 2)

                # Check for completion
                if (abs(error) < error_threshold) or (np.sum(column_mask) / area > 0.7):
                    self.stop_motor()
                    break

                # PD control for turning
                turn_speed = Kp * error + Kd * (error - last_error)
                last_error = error
                self.set_robot_velocity(forward_speed + turn_speed, forward_speed - turn_speed)
                self.step(self.time_step)
            else: # No red detected
                self.stop_motor()
                break

    def center_column_in_view(self, color):

        # PD controller constants
        Kp = ALIGN_COLUMN_KP
        Kd = ALIGN_COLUMN_KD

        error_threshold = ALIGN_COLUMN_ERROR_THRESHOLD
        last_error = 0
        align_step_count = 0

        while self.step(self.time_step) != -1:
            align_step_count += 1
            if align_step_count > 100:
                self.stop_motor()
                break

            hsv_img = self.get_hsv_image()
            if hsv_img is None:
                self.stop_motor()
                break

            height, width, _ = hsv_img.shape
            column_mask = utils.segment_color(hsv_img, color)
            M = cv2.moments(column_mask)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                error = cX - (width // 2)

                # Check for completion
                if abs(error) < error_threshold:
                    self.stop_motor()
                    break

                # PD control for turning
                turn_speed = Kp * error + Kd * (error - last_error)
                last_error = error
                self.set_robot_velocity(turn_speed, -turn_speed)
                self.step(self.time_step)
            else:  # No column detected
                self.stop_motor()
                break


    def lidar_update_map(self):
        points = self.get_pointcloud_world_coordinates()
        map_position = self.get_map_position()
        self.map_object.lidar_update_grid_map(map_position, points)

    def found_all_2_columns(self):
        if (self.start_point is not None) and (self.end_point is not None):
            return True
        return False
    
    # def slowly_360(self):
    #     self.set_robot_velocity(3, -3)
    #     self.step(8000)
    #     self.stop_motor()

    def slowly_360(self):
        print("[Scan] Starting 360 rotation...")
        self.set_robot_velocity(3, -3)
        steps_taken = 0
        target_steps = 8500 // self.time_step
        
        while steps_taken < target_steps:
            if self.step(self.time_step) == -1:
                break
            steps_taken += 1
            
            # Check the thread signal
            with self.detection_lock:
                if self.camera_detection_signal is not None:
                    # If we see a column, STOP immediately
                    if isinstance(self.camera_detection_signal, tuple) and self.camera_detection_signal[0] == 'column':
                        print(f"---- Column {self.camera_detection_signal[1]} detected! Stopping 360 ---")
                        break
                    # Also stop for other important detections
                    elif self.camera_detection_signal in ['red_wall', 'green_carpet']:
                        print(f"---- {self.camera_detection_signal} detected! Stopping 360 ---")
                        break
        
        self.stop_motor()
            

    def handle_frontier_exploration(self, count, map_diff):
        frontier_regions = []
        chosen_frontier = None
        path_to_frontier = None
        chasing_column = False

        if count >= EXPLORATION_START_FRONTIER_AFTER and count % EXPLORATION_FRONTIER_SELECTION_FREQ == 0:
            frontier_regions = self.map_object.detect_frontiers()

            # Occasionally bias frontier choice near known column estimates/start/end
            if random.random() < 0.9:
                chosen_frontier = self.select_random_frontier_near_column()
                chasing_column = (chosen_frontier is not None)
                print("----Frontire near column---")

            # Fallback to existing selection logic if none chosen
            if chosen_frontier is None:
                if random.random() < 0.4:
                    chosen_frontier = self.select_frontier_target(frontier_regions)
                    print(count, "--==--Nearest frontier---")
                    self.chosen_frontier_count += 1
                else:
                    chosen_frontier = self.select_frontier_target2(frontier_regions)
                    print(count, "===Random frontier---")
                    self.chosen_frontier_count = 0

            if chosen_frontier:
                path_to_frontier = self.map_object.find_path_for_frontier(self.get_map_position(), chosen_frontier)
                if path_to_frontier:
                    success = self.frontier_following(path_to_frontier)
                    if success and chasing_column and (self.start_point is None or self.end_point is None):
                        self.slowly_360()
                        

        return frontier_regions, chosen_frontier, path_to_frontier
    
    def mark_column(self, color):
        wp = self.position_ahead(0.2)  # 0.2 meters ahead
        mp = self.convert_to_map_coordinates(wp[0], wp[1])

        if color == 'blue':
            self.start_point = mp
        elif color == 'yellow':
            self.end_point = mp

        # track last found
        self.last_found_color = color
        self.last_found_point = mp

    def detect_column_strict(self, min_pixels=800):
        hsv = self.get_hsv_image()
        if hsv is None:
            return None

        # optional: use bottom half (more stable close-up)
        h = hsv.shape[0]
        hsv_roi = hsv[h//2:, :]

        yellow_mask = utils.segment_color(hsv_roi, 'yellow')
        if cv2.countNonZero(yellow_mask) > min_pixels:
            return 'yellow'

        blue_mask = utils.segment_color(hsv_roi, 'blue')
        if cv2.countNonZero(blue_mask) > min_pixels:
            return 'blue'

        return None

    # def mark_column(self, color):
    #     if color == 'blue':
    #         self.start_point = self.get_map_position() 
    #     elif color == 'yellow':
    #         self.end_point = self.get_map_position()

    def update_column_estimation_from_view(self, color):
        """Passive estimation update (no steering). Safe to call during path following."""
        print("==---=== column while path following===---==")
        dist = self.estimate_column_distance(color)
        if dist is None:
            return
        wp = self.position_ahead(dist / 100.0)
        mp = self.convert_to_map_coordinates(wp[0], wp[1])
        self.update_column_estimation(color, mp)
        if color == 'blue':
            self.blue_pos_update_count += 1
        else:
            self.yellow_pos_update_count += 1

    def update_column_estimation(self, color, position):
        """
        Only use when the column is far away
        Update estimated position of the column
        Used for estimation only, not the actual marking on the map 
        """
        if color == 'blue':
            if self.blue_estimated_pos is None:
                self.blue_estimated_pos = position
            else:    
                self.blue_estimated_pos = 0.3 * np.array(self.blue_estimated_pos) + 0.7 * np.array(position)  
        if color == 'yellow':
            if self.yellow_estimated_pos is None:
                self.yellow_estimated_pos = position
            else:
                self.yellow_estimated_pos = 0.3 * np.array(self.yellow_estimated_pos) + 0.7 * np.array(position)

    def explore(self, debug=True, keep_threads=True):
        '''
        1. Find blue
        2. Find yellow
        3. Explore randomly, occasionally follow frontier targets

        TODO:
        - improve column distance estimation (far or close)
        - Path following from start to end in another function
        '''

        # Start continuous detection and lidar threads before any setup
        self.start_camera_thread()
        self.start_lidar_thread()
        # self.start_stuck_thread()
        active_frontier = None
        active_path = None


        map_object = self.map_object
        
        # Start visualization in separate thread
        if debug:
            map_object.start_visualization()

        count = 0
        chosen_frontier = None  # store the currently selected frontier
        # planning results/state holders - initialize to avoid stale values between iterations
        path_to_frontier = None
        previous_map = map_object.grid_map.copy()
        map_diff = 1.0
        frontier_regions = []
        last_position = self.get_position()

        #  only for the first time delay 1 sec
        time.sleep(0.2)

        self.slowly_360()

        while self.step(self.time_step) != -1 and not self.found_all_2_columns():

            # Check for camera detection signals from background thread
            with self.detection_lock:
                if self.camera_detection_signal is not None:
                    signal = self.camera_detection_signal
                    self.camera_detection_signal = None  # Reset signal to allow camera thread to set new ones

                    if signal == 'red_wall':
                        self.stop_motor()

                        # 1) Align
                        self.align_to_red_wall()
                        self.stop_motor()

                        # 2) Wait until fully stable (not turning)
                        for _ in range(10):
                            if not self.is_turning():
                                break
                            if self.step(self.time_step) == -1:
                                break

                        # 3) Mark closure
                        ok = self.mark_closure_block()

                        # 4) Small delay so marking appears before moving
                        for _ in range(5):
                            if self.step(self.time_step) == -1:
                                break

                        # 5) Move away
                        self.turn_right_milisecond(350)

                        # ✅ DROP CURRENT FRONTIER/PATH so we don't retry it next loop
                        active_path = None
                        active_frontier = None
                        chosen_frontier = None
                        path_to_frontier = None

                        continue

                        
                        # # print('Done align to red wall')
                        # random_duration = random.randint(500, 700)
                        # self.turn_right_milisecond(random_duration)
                    
            # # --- Periodic green-carpet marking (low-frequency, with cooldown) ---
            # if count % 200 == 0:
            #     try:
            #         # Use the larger min_pixel_threshold to ensure robot is close enough
            #         marked = self.mark_green_carpet_permanently(min_pixel_threshold=10)
            #         if marked:
            #             # If a mark was made, update map with lidar points to protect new area
            #             try:
            #                 points = self.get_pointcloud_world_coordinates()
            #                 map_points = self.convert_to_map_coordinate_matrix(points)
            #                 # If GridMap API exists, call map update helpers
            #                 if hasattr(self, 'bresenham_to_obstacle_score'):
            #                     self.bresenham_to_obstacle_score(map_points)
            #                 if hasattr(self.map_object, 'update_grid_map'):
            #                     self.map_object.update_grid_map()
            #             except Exception:
            #                 pass
            #     except Exception as e:
            #         print(f"[Green Carpet] marking attempt failed: {e}")

            map_diff = utils.percentage_map_differences(previous_map, map_object.grid_map)
            frontier_regions, chosen_frontier, path_to_frontier = self.handle_frontier_exploration(count, map_diff)

            # Fallback: no frontier/path -> pick random nearby freespace
            if path_to_frontier is None:
                if random.random() < 0.2:
                    fallback_frontier = self.select_random_freespace_near_robot()
                    if fallback_frontier is not None:
                        chosen_frontier = fallback_frontier
                        path_to_frontier = self.map_object.find_path_for_frontier(
                            self.get_map_position(),
                            chosen_frontier
                        )

            # select only when nothing active
            if active_path is None and path_to_frontier:
                active_path = path_to_frontier
                active_frontier = chosen_frontier

            # follow if active
            if active_path is not None:
                success = self.frontier_following(active_path)
                active_path = None
                active_frontier = None
            previous_map = map_object.grid_map.copy()

            if debug:
                # Update GridMap state for visualization thread
                with map_object.vis_lock:
                    rx, ry = self.get_map_position()
                    map_object.robot_position = (rx, ry)
                    map_object.current_path = active_path if active_path is not None else path_to_frontier
                    map_object.target_position = active_frontier if active_frontier is not None else chosen_frontier

                    
                    # Update column points for visualization
                    column_points = []
                    if self.blue_estimated_pos is not None:
                        column_points.append((int(self.blue_estimated_pos[0]), int(self.blue_estimated_pos[1]), (0, 255, 255)))
                    if self.yellow_estimated_pos is not None:
                        column_points.append((int(self.yellow_estimated_pos[0]), int(self.yellow_estimated_pos[1]), (255, 255, 0)))
                    map_object.column_points = column_points

            count += 1

        # Clean up detection/lidar threads and visualization
        # If `keep_threads` is True we will reuse the running threads/visualizer
        self.stop_camera_thread()
        self.stop_lidar_thread()
        # self.stop_stuck_thread()
        map_object.stop_visualization()

        # Reuse threads: clear transient signals and transient flags so
        # final path following isn't immediately interrupted by stale state.
        with self.detection_lock:
            self.camera_detection_signal = None
            self.green_carpet_active = False
            self.last_green_carpet_points = []
            time.sleep(0.5)

        self.stop_motor()
        print("Exploration completed.")

        # Build the main path once both columns are found
        if self.start_point is not None and self.end_point is not None:
            blue  = tuple(self.start_point)
            yellow = tuple(self.end_point)

            # later found becomes START
            if self.last_found_color == 'blue':
                start = blue
                goal = yellow
            elif self.last_found_color == 'yellow':
                start = yellow
                goal = blue
            else:
                start = blue
                goal = yellow

            main_path = self.find_path(start, goal)

        else:
            main_path = []

        return main_path
    
        # print("Exploration completed.")
        # return self.path


    def find_path(self, start_point, end_point):
        """Delegate path finding to GridMap manager."""
        return self.map_object.find_path(start_point, end_point)

    def find_path_for_frontier(self, start_point, end_point):
        """Delegate frontier path finding to GridMap manager."""
        return self.map_object.find_path_for_frontier(start_point, end_point)


    def align_to_path(self, map_target, angle_threshold_deg=15, clear_distance=0.6, back_distance=0.18):
        """Align robot to face a map target before following a planned path.

        Behavior:
        - Compute heading to the `map_target`.
        - If angular error > `angle_threshold_deg`:
            - If front clearance > `clear_distance`: rotate in place slowly.
            - Else: back up by `back_distance`, then rotate in place.
        Returns True when alignment is completed or not needed.
        """
        if map_target is None:
            return True

        # Convert map target to world coordinates
        try:
            tx_w, ty_w = self.convert_to_world_coordinates(map_target[0], map_target[1])
        except Exception:
            return True

        rx, ry = self.get_position()
        target_angle = np.arctan2(ty_w - ry, tx_w - rx)
        heading = self.get_heading('rad')

        # normalize angle difference to [-pi, pi]
        angle = target_angle - heading
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi

        abs_deg = abs(np.degrees(angle))
        if abs_deg < angle_threshold_deg:
            return True

        # Decide whether to back up first
        front_min = self.get_min_front_distance()

        # backing parameters
        back_speed = 0.12  # m/s (approx)
        dt = TIME_STEP / 1000.0

        if front_min <= clear_distance:
            # compute steps to back up approximately back_distance
            if back_distance > 0:
                step_count = max(1, int(back_distance / (abs(back_speed) * dt)))
                left_w, right_w = self.velocity_to_wheel_speeds(-back_speed, 0.0)
                self.set_robot_velocity(left_w, right_w)
                for _ in range(step_count):
                    if self.step(self.time_step) == -1:
                        break
                self.stop_motor()

        # rotate in place slowly to face the target
        # target absolute rotation in radians
        rad_to_turn = abs(angle)
        # pick a moderate angular speed (rad/s)
        w = 1.0
        # choose direction so wheel speeds have correct sign
        if angle > 0:
            w_cmd = w
        else:
            w_cmd = -w

        left_speed, right_speed = self.velocity_to_wheel_speeds(0.0, w_cmd)
        # start rotation
        self.set_robot_velocity(left_speed, right_speed)

        initial_heading = self.get_heading('rad')
        while self.step(self.time_step) != -1:
            current = self.get_heading('rad')
            turned = current - initial_heading
            # normalize
            while turned > np.pi:
                turned -= 2 * np.pi
            while turned < -np.pi:
                turned += 2 * np.pi
            if abs(turned) >= rad_to_turn - 0.03:
                break

        self.stop_motor()
        return True

    # def frontier_following(self, path, vis=None):
    #     stuck_counter = 0
    #     for target in path[5::5]:
    #         while self.step() != -1 and self.camera_detection_signal is None:
    #             last_position = self.get_position()
    #             for event in pygame.event.get(): 
    #                 if event.type == pygame.QUIT:
    #                     exit()

    #             if vis:
    #                 vis.display(self.map_object.grid_map)
    #                 vis.draw_path(path)
    #                 rx, ry = self.get_map_position()
    #                 vis.draw_point(rx, ry, color=(0, 0, 255), radius=5)
    #                 vis.draw_point(target[0], target[1], color=(0, 255, 0), radius=3)
    #                 pygame.display.flip()

                
                
    #             if self.follow_local_target(target):
    #                 stuck_counter = 0
    #                 break
                
    #             if self.robot_stuck(last_position, stuck_distance=0.08):
    #                 stuck_counter += 1
    #                 if stuck_counter > 50:
    #                     print("Stuck in frontier_following, drop path...")
    #                     self.stop_motor()
    #                     # self.lidar_update_map()
    #                     self.recover_from_stuck()
                        
    #                     return
    #     self.stop_motor()

    def frontier_following(self, path, vis=None, replan_interval=20, drop_on_red_wall=True, use_global_planner=False):

        """
        Frontier following with:
        - timestep-based replanning
        - front obstacle recovery (reverse → wait → replan)
        - ACTIVE red wall handling (interrupt + closure + align + turn)
        - ACTIVE green carpet marking (polled periodically; does NOT drop path unless you want it to)
        - PASSIVE column handling (observe/mark only; never drop path)
        """

        if path is None or len(path) == 0:
            return False

        frontier_goal = path[-1]          # Fixed goal
        current_path = list(path)
        timestep_counter = 0
        stuck_counter = 0
        stuck_attempt_count = 0     # Track total stuck events (not replan failures)
        #
        replan_attempts = 0  # Track how many times we've replanned due to obstacles

        target_index = 5
        OBSTACLE_THRESHOLD = 0.08   # meters
        REVERSE_DISTANCE = 0.25     # meters
        MAP_UPDATE_WAIT_STEPS = 40  # allow LiDAR/map to update
        CARPET_CHECK_EVERY = 10     # timesteps (cheap + safe)
        MAX_STUCK_ATTEMPTS = 3      # maximum stuck events before dropping frontier
        distance_to_prev = 99.0
        prev_position = self.get_position()

        while target_index < len(current_path):
            target = current_path[target_index]

            while self.step(self.time_step) != -1:
                timestep_counter += 1
                if self.interrupt_path:
                    print("[Frontier] Path interrupted by column marking!")
                    self.stop_motor()
                    self.interrupt_path = False  # Reset the flag
                    return True  # Return True because we found what we wanted
                last_position = self.get_position()
                self.counter_obstacle_recoveries = 0

                # --------------------------------------------------
                # 1) ACTIVE: Handle camera signals here (selective)
                # --------------------------------------------------
                signal = None
                with self.detection_lock:
                    if self.camera_detection_signal is not None:
                        signal = self.camera_detection_signal
                        self.camera_detection_signal = None

                # 3) FRONT OBSTACLE → REVERSE → REPLAN (Max 3 tries)
                # --------------------------------------------------
                if self.obstacle_in_front():
                    replan_attempts += 1
                    print(f"[Frontier] Obstacle detected! Attempt {replan_attempts}/2")
                    
                    self.recover_from_obstacle() # Back up
                    self.lidar_update_map()      # Update map with the new obstacle

                    if replan_attempts >= 2:
                        print("[Frontier] Failed 2 times. Giving up on this frontier for now.")
                        # NOTE: We do NOT mark it as visited here, so it stays in the pool
                        return False

                    # Try to replan to the same goal
                    try:
                        current_start = self.get_map_position()
                        new_path = self.map_object.find_path_for_frontier(current_start, frontier_goal)
                        if new_path and len(new_path) > 5:
                            print("[Frontier] Replanned successfully! Trying new route...")
                            current_path = new_path
                            target_index = 5
                            break # Exit inner loop to start following the new path
                        else:
                            print("[Frontier] No path found after obstacle. Dropping.")
                            return False
                    except Exception as e:
                        print(f"[Frontier] Replan error: {e}")
                        return False


                # 🔴 Red wall: INTERRUPT path following (same as explore)
                if signal == 'red_wall':
                    print("[Frontier] Red wall detected during path following")

                    self.stop_motor()

                    # 1) Align
                    self.align_to_red_wall()
                    self.stop_motor()

                    # 2) Wait until fully stable (not turning)
                    for _ in range(10):
                        if not self.is_turning():
                            break
                        if self.step(self.time_step) == -1:
                            break

                    # 3) Mark closure
                    ok = self.mark_closure_block()
                    print(f"[info] closure_mark_ok={ok}")

                    # 4) Small delay so marking appears before moving
                    for _ in range(10):
                        if self.step(self.time_step) == -1:
                            break
                    
                    # 5) Move away a bit
                    self.turn_right_milisecond(350)

                    if drop_on_red_wall:
                        if not hasattr(self.map_object, "visited_frontiers"):
                            self.map_object.visited_frontiers = []
                        self.map_object.visited_frontiers.append(tuple(frontier_goal))
                        self.stop_motor()
                        return False

                  # 🔵🟡 Column: PASSIVE only (consume signal; keep moving)
                

                # --------------------------------------------------
                # 2) ACTIVE: Green carpet marking during path following
                #    (Polling approach: safe, uses your internal guards)
                # --------------------------------------------------
                if signal == 'green_carpet' or timestep_counter % CARPET_CHECK_EVERY == 0:
                    self.stop_motor()
                    # time.sleep(0.5)
                    
                    marked = self.mark_green_carpet_permanently(min_pixel_threshold=10000)
                    if marked:
                        self.stop_motor()
                        time.sleep(0.5)

                        return False

                if isinstance(signal, tuple) and len(signal) == 2:
                    _, color = signal
                    current_pos = self.get_position()
                    prev_pos = self.blue_prev_estimate_position if color == 'blue' else self.yellow_prev_estimate_position
                    
                    # Check distance from previous estimation position
                    if prev_pos is None:
                        distance_to_prev = float('inf')
                    else:
                        distance_to_prev = float(np.linalg.norm(current_pos - np.array(prev_pos)))
                    
                    # Primary condition: distance >= threshold
                    if distance_to_prev >= 0.8:
                        self.stop_motor()
                        self.center_column_in_view(color)
                        height_pixels = self.get_column_center_pixels(color) 
                        time.sleep(0.1)
                        if height_pixels >= 8:
                            self.update_column_estimation_from_view(color)
                        # Update the prev position after centering (outside ratio check so it always runs)
                        if color == 'blue':
                            self.blue_prev_estimate_position = current_pos
                        else:
                            self.yellow_prev_estimate_position = current_pos

                # 3) FRONT OBSTACLE → REVERSE → WAIT → DROP FRONTIER
                # --------------------------------------------------
                if self.obstacle_in_front():
                    print("[Frontier] Virtual Bumper triggered! Recovering...")
                    self.recover_from_obstacle()
                    self.lidar_update_map()
                    
                    # Mark frontier as visited to avoid retrying the same blocked path
                    if not hasattr(self.map_object, "visited_frontiers"):
                        self.map_object.visited_frontiers = []
                    self.map_object.visited_frontiers.append(tuple(frontier_goal))
                    
                    return False
                
                    # replan with updated map
                    # try:
                    #     current_start = self.get_map_position()
                    #     new_path = self.map_object.find_path_for_frontier(current_start, frontier_goal)
                    #     if new_path:
                    #         print("[Frontier] Replanned after obstacle")
                    #         current_path = new_path
                    #         target_index = 5
                    #         break
                    #     else:
                    #         print("[Frontier] Replan failed, dropping frontier")
                    #         return False
                    # except Exception as e:
                    #     print(f"[Frontier] Replan error: {e}")
                    #     return False

                # --------------------------------------------------
                # 4) PERIODIC TIMESTEP-BASED REPLANNING
                # --------------------------------------------------
                if timestep_counter % replan_interval == 0:
                    try:
                        current_start = self.get_map_position()
                        new_path = self.map_object.find_path_for_frontier(current_start, frontier_goal)
                        if new_path and len(new_path) > 5:
                            print("[Replan] Periodic path update")
                            current_path = new_path
                            target_index = 5
                            break
                    except Exception as e:
                        print(f"[Replan] Failed: {e}")

                # --- Visualization update (GridMap thread draws these) ---
                with self.map_object.vis_lock:
                    rx, ry = self.get_map_position()
                    self.map_object.robot_position = (int(rx), int(ry))
                    self.map_object.current_path = current_path   # show live (replanned) path
                    self.map_object.target_position = target      # show current waypoint


                # --------------------------------------------------
                # 6) Follow local target
                # --------------------------------------------------
                reached, is_stuck = self.follow_local_target(target)
                if is_stuck:
                    stuck_attempt_count += 1
                    
                    # Check if we've exceeded max stuck attempts
                    if stuck_attempt_count >= MAX_STUCK_ATTEMPTS:
                        if not hasattr(self.map_object, "visited_frontiers"):
                            self.map_object.visited_frontiers = []
                        self.map_object.visited_frontiers.append(tuple(frontier_goal))
                        self.stop_motor()
                        return False
                    
                    self.stop_motor()

                    # 1) recover
                    self.lidar_update_map()
                    self.recover_from_stuck()

                    # 2) replan to SAME frontier_goal
                    try:
                        current_start = self.get_map_position()
                        if use_global_planner:
                            new_path = self.map_object.find_path(tuple(current_start), tuple(frontier_goal))
                        else:
                            new_path = self.map_object.find_path_for_frontier(current_start, frontier_goal)

                        if new_path and len(new_path) > 5:
                            current_path = list(new_path)
                            target_index = 5
                            break  # exit inner while, outer loop will pick new target from updated path
                        else:
                            if not hasattr(self.map_object, "visited_frontiers"):
                                self.map_object.visited_frontiers = []
                            self.map_object.visited_frontiers.append(tuple(frontier_goal))
                            return False
                            
                    except Exception as e:
                        print(f"[Frontier] Replan error: {e}, dropping frontier")
                        if not hasattr(self.map_object, "visited_frontiers"):
                            self.map_object.visited_frontiers = []
                        self.map_object.visited_frontiers.append(tuple(frontier_goal))
                        return False
                # if is_stuck:
                #     self.stop_motor()
                #     self.turn_right_milisecond(random.randint(200, 400))
                #     self.move_backward_milisecond()
                #     # return False
                if reached:
                    break

            target_index += 5

        self.stop_motor()
        return True

    def follow_final_path(self, path, debug_vis=False, replan_interval=60):
        """
        Dedicated final path follower (blue <-> yellow), frontier-like behavior but:
        - uses GLOBAL planner (GridMap.find_path) for every replan
        - never falsely returns True on short paths
        - restarts camera/lidar threads (explore() stops them)
        - follows intermediate local targets (waypoints) and replans when needed
        """

        if path is None or len(path) == 0:
            print("[FinalPath] Empty path")
            return False

        # Ensure background threads are running (explore() stops them)
        if not self.camera_thread_running:
            self.start_camera_thread()
        if not self.lidar_thread_running:
            self.start_lidar_thread()
            print("[FinalPath] Restarted lidar thread")

        if debug_vis:
            try:
                self.map_object.start_visualization()
            except Exception:
                pass

        # Fixed final goal = last waypoint of the planned path
        goal = tuple(path[-1])

        # If we're already basically at goal, finish cleanly
        if self.get_map_distance(goal) < PATH_FOLLOWING_TARGET_REACH_DISTANCE:
            self.stop_motor()
            print("[FinalPath] Already at goal")
            return True

        def plan_from_current_to_goal():
            start = tuple(self.get_map_position())
            try:
                new_path = self.map_object.find_path(start, goal)
            except Exception as e:
                print(f"[FinalPath] Planner error: {e}")
                return None
            return new_path

        # Start from a fresh plan from CURRENT position (not necessarily path[0])
        current_path = plan_from_current_to_goal()
        if not current_path or len(current_path) < 2:
            # Fallback to given path if planner fails (better than nothing)
            current_path = list(path)

        # Optional: align to first meaningful waypoint
        try:
            first_wp = current_path[min(1, len(current_path) - 1)]
            self.align_to_path(first_wp)
        except Exception:
            pass

        def build_targets(p):
            """Pick intermediate waypoints like frontier-following, always include final goal."""
            p = list(p)
            if len(p) <= 1:
                return [goal]

            # dynamic stride: smaller stride for shorter paths
            if len(p) < 25:
                stride = 3
            elif len(p) < 60:
                stride = 4
            else:
                stride = 5

            idxs = list(range(stride, len(p), stride))
            if len(p) - 1 not in idxs:
                idxs.append(len(p) - 1)

            # Safety: if stride overshot for some reason
            idxs = [i for i in idxs if 0 <= i < len(p)]
            if not idxs:
                idxs = [len(p) - 1]

            return [tuple(p[i]) for i in idxs]

        targets = build_targets(current_path)

        # Tunables (kept close to your frontier_following behavior)
        timestep_counter = 0
        OBSTACLE_THRESHOLD = 0.08    # meters
        REVERSE_DISTANCE = 0.18      # meters
        MAP_UPDATE_WAIT_STEPS = 40   # steps
        CARPET_CHECK_EVERY = 10      # steps

        i = 0
        while i < len(targets):
            target = targets[i]

            while self.step(self.time_step) != -1:
                timestep_counter += 1

                # -----------------------------
                # (1) Handle camera signals
                # -----------------------------
                signal = None
                with self.detection_lock:
                    if self.camera_detection_signal is not None:
                        signal = self.camera_detection_signal
                        self.camera_detection_signal = None

                # Red wall (final run): align + mark closure + replan to same goal
                if signal == "red_wall":
                    print("[FinalPath] Red wall detected during final following")
                    self.stop_motor()

                    try:
                        self.align_to_red_wall()
                        self.stop_motor()
                    except Exception:
                        pass

                    # Wait until stable
                    for _ in range(10):
                        if not self.is_turning():
                            break
                        if self.step(self.time_step) == -1:
                            break

                    # Mark closure
                    try:
                        ok = self.mark_closure_block()
                        print(f"[FinalPath] closure_mark_ok={ok}")
                    except Exception as e:
                        print(f"[FinalPath] closure mark failed: {e}")

                    # Small delay to let it “stick”
                    for _ in range(10):
                        if self.step(self.time_step) == -1:
                            break

                    # Move away a bit
                    try:
                        self.turn_right_milisecond(350)
                    except Exception:
                        pass

                    # Wait for lidar thread / map update
                    for _ in range(MAP_UPDATE_WAIT_STEPS):
                        if self.step(self.time_step) == -1:
                            self.stop_motor()
                            return False

                    # Replan and restart target list
                    new_path = plan_from_current_to_goal()
                    if new_path and len(new_path) > 2:
                        current_path = list(new_path)
                        targets = build_targets(current_path)
                        i = 0
                        target = targets[i]
                        break
                    else:
                        print("[FinalPath] Replan failed after red wall")
                        self.stop_motor()
                        return False

                # Column signals: active mark if very close, otherwise passive estimation
                if isinstance(signal, tuple) and len(signal) >= 2 and signal[0] == "column":
                    _, color = signal
                    try:
                        center_ratio = 0.0
                        try:
                            center_ratio = self.get_column_center_ratio(color)
                        except Exception:
                            pass

                        depth_close = False
                        try:
                            d = self.estimate_column_distance(color)
                            if d is not None and d < COLOR_DETECTION_DEPTH_THRESHOLD:
                                depth_close = True
                        except Exception:
                            pass

                        lidar_close = False
                        try:
                            if self.get_min_front_distance() < 0.35:
                                lidar_close = True
                        except Exception:
                            pass

                        if center_ratio >= 0.20 or depth_close or lidar_close:
                            print(f"[Column] Active mark (final) triggered: color={color}, center_ratio={center_ratio:.3f}, depth_close={depth_close}, lidar_close={lidar_close}")
                            self.stop_motor()
                            for _ in range(10):
                                if not self.is_turning():
                                    break
                                if self.step(self.time_step) == -1:
                                    break

                            try:
                                self.center_column_in_view(color)
                            except Exception as e:
                                print(f"[Column] center_column_in_view failed (final): {e}")

                            try:
                                dist = self.estimate_column_distance(color)
                            except Exception as e:
                                print(f"[Column] estimate_column_distance failed (final): {e}")
                                dist = None

                            try:
                                if dist is not None:
                                    self.mark_on_map(dist, color=color)
                                else:
                                    if center_ratio >= 0.20:
                                        self.mark_column(color)
                                print(f"[FinalPath] Column {color} marked. Breaking path to re-evaluate.")
                                self.stop_motor()

                                continue
                            except Exception as e:
                                print(f"[Column] mark failed (final): {e}")

                            for _ in range(5):
                                if self.step(self.time_step) == -1:
                                    break
                        else:
                            try:
                                self.update_column_estimation_from_view(color)
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"[Column] Column signal handling error (final): {e}")

                # -----------------------------
                # (2) Green carpet marking (optional)
                # -----------------------------
                if timestep_counter % CARPET_CHECK_EVERY == 0:
                    try:
                        self.mark_green_carpet_permanently(min_pixel_threshold=10000)
                    except Exception:
                        pass

                # -----------------------------
                # (3) Front obstacle -> reverse -> wait -> replan
                # -----------------------------
                front_dist = self.get_min_front_distance()
                if front_dist < OBSTACLE_THRESHOLD:
                    print(f"[FinalPath] Obstacle ahead at {front_dist:.2f} m -> reverse + replan")
                    self.stop_motor()

                    back_speed = 0.12  # m/s
                    dt = (TIME_STEP + 80) / 1000.0
                    steps = max(1, int(REVERSE_DISTANCE / (back_speed * dt)))

                    lw, rw = self.velocity_to_wheel_speeds(-back_speed, 0.0)
                    self.set_robot_velocity(lw, rw)
                    for _ in range(steps):
                        if self.step(self.time_step) == -1:
                            break
                    self.stop_motor()

                    for _ in range(MAP_UPDATE_WAIT_STEPS):
                        if self.step(self.time_step) == -1:
                            break

                    new_path = plan_from_current_to_goal()
                    if new_path and len(new_path) > 2:
                        current_path = list(new_path)
                        targets = build_targets(current_path)
                        i = 0
                        target = targets[i]
                        break
                    else:
                        print("[FinalPath] Replan failed after obstacle")
                        self.stop_motor()
                        return False

                # -----------------------------
                # (4) Periodic replan
                # -----------------------------
                if replan_interval and (timestep_counter % replan_interval == 0):
                    new_path = plan_from_current_to_goal()
                    if new_path and len(new_path) > 2:
                        print("[FinalPath] Periodic replan update")
                        current_path = list(new_path)
                        targets = build_targets(current_path)
                        i = 0
                        target = targets[i]
                        break

                # -----------------------------
                # (5) Visualization state update
                # -----------------------------
                try:
                    with self.map_object.vis_lock:
                        rx, ry = self.get_map_position()
                        self.map_object.robot_position = (int(rx), int(ry))
                        self.map_object.current_path = current_path
                        self.map_object.target_position = target
                except Exception:
                    pass

                # -----------------------------
                # (6) Follow local target (DWA)
                # -----------------------------
                reached, is_stuck = self.follow_local_target(target)

                # If stuck, try a recovery + replan (instead of instantly quitting)
                if is_stuck or (min(self.get_distances()) < 0.05):
                    print("[FinalPath] Local following stuck -> recover + replan")
                    self.stop_motor()
                    try:
                        self.recover_from_stuck()
                    except Exception:
                        pass

                    for _ in range(MAP_UPDATE_WAIT_STEPS):
                        if self.step(self.time_step) == -1:
                            break

                    new_path = plan_from_current_to_goal()
                    if new_path and len(new_path) > 2:
                        current_path = list(new_path)
                        targets = build_targets(current_path)
                        i = 0
                        target = targets[i]
                        break
                    else:
                        self.stop_motor()
                        return False

                if reached:
                    # If we reached the final waypoint (or are close to goal), finish
                    if self.get_map_distance(goal) < PATH_FOLLOWING_TARGET_REACH_DISTANCE:
                        self.stop_motor()
                        print("[FinalPath] Goal reached")
                        return True
                    break

            # advance to next local target
            i += 1

            # final safety check
            if self.get_map_distance(goal) < PATH_FOLLOWING_TARGET_REACH_DISTANCE:
                self.stop_motor()
                print("[FinalPath] Goal reached (post-check)")
                return True

        self.stop_motor()
        success = (self.get_map_distance(goal) < PATH_FOLLOWING_TARGET_REACH_DISTANCE)
        print(f"[FinalPath] Finished, success={success}")
        return success



    def final_goal_following(self, path):
        # final run: do NOT drop on red wall, and use the global planner when replanning
        return self.frontier_following(path, replan_interval=60, drop_on_red_wall=False, use_global_planner=True)





    def path_following_pipeline(self, path, frontier_target=None):
        """Follow a path with visualization through GridMap state.
        
        Shows the path overlaid on the occupancy grid map via the visualization thread.
        """
        # Smooth the path using spline A* to get smoother waypoints
        try:
            from astar_2_spline import runAStarSearch as runAStarSearchSpline
            if len(path) > 1:
                smoothed_path = runAStarSearchSpline(self.map_object.grid_map, path[0], path[-1])
                if smoothed_path and len(smoothed_path) > 1:
                    path = smoothed_path
                    print(f'[info] Path smoothed with spline A*: {len(path)} waypoints')
        except Exception as e:
            print(f'[warning] Path smoothing failed: {e}, using original path')
        
        if len(path) > 30:
            path = path[::4]
        # Try to align to the first path waypoint before moving to avoid overshoot
        if len(path) > 0:
            try:
                self.align_to_path(path[0])
            except Exception:
                pass
        count = 0
        stuck_counter = 0
        last_position = self.get_position()

        # sample targets, but ALWAYS include the final waypoint
        sampled_targets = list(path[::10]) if len(path) > 10 else list(path)

        if len(sampled_targets) == 0:
            self.stop_motor()
            return False

        if sampled_targets[-1] != path[-1]:
            sampled_targets.append(path[-1])


        for idx, target in enumerate(sampled_targets):
            stuck_counter = 0  # Reset stuck counter for each target
            while self.step() != -1:
                # Get front distance to wall for monitoring
                front_distance = self.get_min_front_distance()

                # Update GridMap state for visualization
                with self.map_object.vis_lock:
                    rx, ry = self.get_map_position()
                    self.map_object.robot_position = (rx, ry)
                    self.map_object.current_path = path
                    self.map_object.target_position = target
                    if frontier_target:
                        # Add frontier target as a special column point
                        self.map_object.column_points = [(frontier_target[0], frontier_target[1], (0, 200, 255))]
                
                # --- Check for wall blocking the path before stuck detection ---
                wall_threshold = 0.2  # meters (20cm) - wall very close
                if front_distance < wall_threshold:
                    print(f'[warning] Wall detected at {front_distance:.2f}m blocking path to target {target}. Skipping frontier.')
                    if frontier_target is not None:
                        self.map_object.visited_frontiers.append(frontier_target)

                    self.stop_motor()
                    return False
                
                # --- Simple stuck detection: if no progress in 50 steps, try recovery ---
                cur_pos = self.get_position()
                dist_moved = float(np.linalg.norm(cur_pos - last_position))
                
                if dist_moved < 0.01:  # Less than 1cm movement
                    stuck_counter += 1
                else:
                    stuck_counter = 0  # Reset if we moved
                    last_position = cur_pos
                
                if stuck_counter > 50:
                    print(f'[info] Path following stuck for 50 steps, recovering')
                    self.recover_from_stuck()
                    stuck_counter = 0
                    last_position = self.get_position()

                reached, is_stuck = self.follow_local_target(target)

                if is_stuck:
                    self.stop_motor()
                    self.recover_from_stuck()
                    break

                if reached:
                    print(f'Reached target: {target} (front distance to wall: {front_distance:.2f}m)')
                    if idx == len(sampled_targets) - 1:
                        self.stop_motor()
                        return True
                    break

                
                count += 1

        # If we exit the loops without confirming final reach, stop motors and report failure
        self.stop_motor()
        return False

    def get_pointcloud_2d(self):
        if self.lidar is None:
            return np.array([])
        
        points = self.lidar.getPointCloud()
        if points is None or len(points) == 0:
            return np.array([])
            
        points = np.array([[point.x, point.y] for point in points])
        points = points[~np.isinf(points).any(axis=1)]
        return points

    def there_is_obstacle(self, map_target):
        """Delegate obstacle check to GridMap manager."""
        return self.map_object.there_is_obstacle(map_target)

    
    def transform_points_to_world(self, points_local):
        """    def path_following_pipeline(self, path, frontier_target=None):
        Transform a batch of 2D points from robot-local frame to world frame using NumPy.

        Parameters:
            points_local: np.ndarray of shape (N, 2) — local [x, y] points

        Returns:
            points_world: np.ndarray of shape (N, 2) — transformed points in world coordinates
        """
        x_robot, y_robot = self.get_position()
        theta = self.get_heading('rad')  # yaw

        # Rotation matrix R(theta)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        # Apply rotation and translation
        points_rotated = points_local @ R.T  # shape (N, 2)
        points_world = points_rotated + np.array([x_robot, y_robot])  # broadcast translation

        return points_world

    def get_pointcloud_world_coordinates(self):
        points_local = self.get_pointcloud_2d()
        points_world = self.transform_points_to_world(points_local)
        return points_world

    def detect_column(self):
        camera_frame = self.get_hsv_image()
        if camera_frame is None:
            return None

        yellow_mask = utils.segment_color(camera_frame, 'yellow')
        if cv2.countNonZero(yellow_mask):
            return 'yellow'

        blue_mask = utils.segment_color(camera_frame, 'blue')
        if cv2.countNonZero(blue_mask):
            return 'blue'

        return None


    def robot_stuck(self, last_position, stuck_distance=0.16):
        cur_position = self.get_position()
        if np.linalg.norm(cur_position - last_position) < stuck_distance:
            return True
        return False
    
    def recover_from_stuck(self, turn_duration=(400, 600)):
        speeds = random.choice([(-8, -8), (-8, -8)])
        self.set_robot_velocity(speeds[0], speeds[1])
        # check if there is enough space to back up
        # checking back clearanc with get_back_clearance_distance()

        distances = self.get_distances()
        if distances[1] > 0.25 and distances[3] > 0.25:
            print("reversing longer")
            self.step(500)
        else:
            print("reversing shorter")
            self.step(250)

        
        
        self.stop_motor()
         # Wait for lidar thread / map update
        for _ in range(40):
            if self.step(self.time_step) == -1:
                self.stop_motor()
                return False

        random_duration = random.randint(turn_duration[0], turn_duration[1])
        # turn randomly left or right
        if random.random() < 0.5:
            self.turn_left_milisecond(random_duration)
        else:
            self.turn_right_milisecond(random_duration)
        self.set_robot_velocity(5, 5)
        self.step(250)

        
        
        self.stop_motor()
         # Wait for lidar thread / map update
        for _ in range(40):
            if self.step(self.time_step) == -1:
                self.stop_motor()
                return False
            
    def recover_from_obstacle(self, turn_duration=(400, 600)):
        speeds = random.choice([(-8, -8), (-8, -8)])
        self.set_robot_velocity(speeds[0], speeds[1])
        # check if there is enough space to back up
        # checking back clearanc with get_back_clearance_distance()

        distances = self.get_distances()
        if distances[2] > 0.25 and distances[3] > 0.25:
            print("reversing longer [recover from obstacle]")
            self.step(500)
        else:
            print("reversing shorter [recover from obstacle]")
            self.step(250)

        
        
        self.stop_motor()
            # Wait for lidar thread / map update
        for _ in range(80):
            if self.step(self.time_step) == -1:
                self.stop_motor()
                return False      
        

        # random_duration = random.randint(turn_duration[0], turn_duration[1])
        # # turn randomly left or right
        # if random.random() < 0.5:
        #     self.turn_left_milisecond(random_duration)
        # else:
        #     self.turn_right_milisecond(random_duration)
        # self.set_robot_velocity(5, 5)
        self.step(250)

       
        # while min(distances[0], distances[2]) < 0.05:
        #     print('Still closed to obstacle')
        #     self.step(20)
        #     time.sleep(2)
        #     distances = self.get_distances()
    

    def plan_and_follow_frontier(self, top_k=3):
        """Select a frontier target, plan a path, and follow it using GridMap manager."""
        # Compute density map
        border_map = utils.boundary_laplacian(self.map_object.grid_map)
        density_map = utils.boundary_density(border_map, r=15)

        # Get top frontier positions
        top_targets = utils.top_frontier_positions(density_map, top_k=top_k)
        
        if top_targets is None or len(top_targets) == 0:
            print('[Frontier] No frontier targets detected in density map')
            return [], None, None

        # Make sure the visited_frontiers list exists
        if not hasattr(self.map_object, "visited_frontiers"):
            self.map_object.visited_frontiers = []
            
        filtered_targets = []
        if len(self.map_object.visited_frontiers) == 0:
            filtered_targets = top_targets
        else:
            filtered_targets = []
            for f in top_targets:
                # Keep frontier if not within 50 px of any previously reached frontier
                if all(np.linalg.norm(np.array(f) - np.array(v)) > 50 for v in self.map_object.visited_frontiers if v is not None):
                    filtered_targets.append(f)

        # If all frontiers are too close to visited ones, try all top_targets
        if len(filtered_targets) == 0:
            print("[Frontier] All frontiers too close to visited ones, trying all candidates")
            filtered_targets = list(top_targets)

        # Get robot position once
        start_pos = self.get_map_position()
        
        # Try each frontier in order until we find a reachable one
        reachable_frontier = None
        path_to_frontier = None
        
        for frontier_candidate in filtered_targets:
            print(f'[Frontier] Attempting frontier at {frontier_candidate}...')
            candidate_path = self.map_object.find_path_for_frontier(start_pos, frontier_candidate)
            
            # If we found a valid path, use this frontier
            if candidate_path and len(candidate_path) > 0:
                reachable_frontier = frontier_candidate
                path_to_frontier = candidate_path
                print(f'[Frontier] Found path to frontier {reachable_frontier} (length: {len(path_to_frontier)})')
                break
            else:
                print(f'[Frontier] No path to frontier {frontier_candidate}, trying next...')
        
        # If no reachable frontier was found
        if reachable_frontier is None:
            print('[Frontier] No reachable frontier found in current frontier set')
            return filtered_targets, None, None
        
        return filtered_targets, path_to_frontier, reachable_frontier

    def select_frontier_target(self, frontier_regions):
        """Selects the best target from clustered frontier regions."""
        if frontier_regions is None:
            return None

        robot_map_pos = self.get_map_position()
        best_region = None
        min_distance = float('inf')

        for region in frontier_regions:
            region_cells = np.array(region)
            distances = np.linalg.norm(region_cells - robot_map_pos, axis=1)
            closest_cell_idx = np.argmin(distances)
            distance_to_region = distances[closest_cell_idx]

            if distance_to_region < min_distance:
                min_distance = distance_to_region
                best_region = region
        
        if best_region is None:
            return None
            
        region_cells = np.array(best_region)
        centroid_x = int(np.mean(region_cells[:, 0]) + random.randint(-5, 5))
        centroid_y = int(np.mean(region_cells[:, 1]) + random.randint(-5, 5))
        
        return (centroid_x, centroid_y)
    
    def select_frontier_target2(self, frontier_regions):
        """Selects a random frontier target from clustered frontier regions."""
        if not frontier_regions:
            return None

        best_region = random.choice(frontier_regions)
            
        region_cells = np.array(best_region)
        centroid_x = int(np.mean(region_cells[:, 0]) + random.randint(-5, 5))
        centroid_y = int(np.mean(region_cells[:, 1]) + random.randint(-5, 5))
        
        return (centroid_x, centroid_y)

    def select_random_frontier_near_column(self, max_jitter=8):
        targets = []
        if self.yellow_estimated_pos is not None and self.end_point is None:
            targets.append(('yellow', self.yellow_estimated_pos))
        if self.blue_estimated_pos is not None and self.start_point is None:
            targets.append(('blue', self.blue_estimated_pos))
            
        if not targets:
            return None
            
        color, pos = random.choice(targets)
        print(f"----{color.upper()} random frontier")
        
        map_h, map_w = self.map_object.grid_map.shape
        
        for _ in range(100): # Limit tries to avoid infinite loop
            jitter_x = random.randint(-max_jitter, max_jitter)
            jitter_y = random.randint(-max_jitter, max_jitter)
            gx = int(pos[0] + jitter_x)
            gy = int(pos[1] + jitter_y)
            
            # Boundary check
            if 0 <= gx < map_w and 0 <= gy < map_h:
                if self.map_object.grid_map[gy, gx] == FREESPACE or self.map_object.grid_map[gy, gx] == UNKNOWN:
                    return (gx, gy)
        
        return None # Fallback if no freespace found nearby

    def select_random_freespace_near_robot(self, radius=40, max_tries=200):
        """Pick a random FREESPACE cell near the robot (map coords)."""
        grid = self.map_object.grid_map
        if grid is None:
            return None

        map_h, map_w = grid.shape
        rx, ry = self.get_map_position()

        for _ in range(max_tries):
            dx = random.randint(-radius, radius)
            dy = random.randint(-radius, radius)
            x = int(rx + dx)
            y = int(ry + dy)

            if x < 0 or x >= map_w or y < 0 or y >= map_h:
                continue

            if grid[y, x] != FREESPACE:
                continue

            if self.map_object.there_is_obstacle((x, y)):
                continue

            return (x, y)

        return None

        # GREEN CARPET LOGIC ---------------------------------------------

    def get_green_carpet_points(self):
        """
        Detects green carpet pixels, projects them to World coordinates (X_world, Y_world)
        on the Z=0 floor plane, using the Z-UP robot frame convention (X-Forward, Y-Lateral).
        
        Returns:
            np.ndarray of shape (N, 2) of [map_x, map_y] coordinates.
        """
        hsv_img = self.get_hsv_image()
        if hsv_img is None:
            return np.array([])

        # --- 1. Color Segmentation and Pixel Selection ---
        # Uses centralized GREEN_HSV_LOWER/UPPER from CONSTANTS.py via utils.segment_color()
        green_mask = utils.segment_color(hsv_img, 'green')
        
        # --- Pixel Expansion using a kernel ---
        kernel_size = GREEN_CARPET_DILATION_KERNEL_SIZE
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        green_mask = cv2.dilate(green_mask, kernel, iterations=GREEN_CARPET_DILATION_ITERATIONS)
        
        h_start = int(self.cam_height * 0.5) 
        green_mask[:h_start, :] = 0
        
        # if cv2.countNonZero(green_mask) < 50:
        #     return np.array([])

        v_pixels, u_pixels = np.where(green_mask == 255) # v=row (y), u=col (x)

        # --- 2. Inverse Pinhole Projection (X-Forward, Z-Up) ---
        
        # A. Normalize pixel coordinates:
        x_norm = (u_pixels - self.cx) / self.fx
        y_norm = (v_pixels - self.cy) / self.fy
        
        # B. Calculate Forward Distance (D = H / y_norm)
        D = self.camera_height_m / (y_norm + 1e-6) 
        
        # Filter: Pixel must be below the horizon (y_norm > 0) AND distance D must be positive and reasonable.
        VALID_DISTANCE_M = 4.0
        valid_mask = (y_norm > 0.001) & (D > 0.1) & (D < VALID_DISTANCE_M)
        
        D_valid = D[valid_mask]
        x_norm_valid = x_norm[valid_mask]
        
        # C. Calculate Local Coordinates relative to the CAMERA origin's ground projection
        X_cam_projected = D_valid 
        Y_cam_projected = -D_valid * x_norm_valid # Check this sign if lateral direction is flipped!
        
        # D. Shift the projected points from the Camera's CoS to the Robot's CoS
        X_offset_local = X_cam_projected + self.X_offset 
        Y_offset_local = Y_cam_projected + self.Y_offset 
        
        # Stack as [X_forward, Y_lateral]
        points_local = np.stack([X_offset_local, Y_offset_local], axis=1) # (N, 2)

        # E & F. Transform to World and Map Coordinates
        points_world = self.transform_points_to_world(points_local)
        green_map_points = self.convert_to_map_coordinate_matrix(points_world)
        
        return green_map_points

    def get_green_carpet_map_mask(self):
        """Return a 2D uint8 mask aligned with `self.grid_map` where detected
        green-carpet map cells are 1 and others 0.
        """
        pts = self.get_green_carpet_points()
        
        if pts.size == 0:
            return np.zeros_like(self.grid_map, dtype=np.uint8)

        # 1. Cast points to integer map indices
        pts = pts.astype(np.int32)
        
        # 2. Filter points outside map boundaries (Vectorized Index Check)
        map_size_y, map_size_x = self.grid_map.shape
        
        valid_x = (pts[:, 0] >= 0) & (pts[:, 0] < map_size_x)
        valid_y = (pts[:, 1] >= 0) & (pts[:, 1] < map_size_y)
        valid_mask = valid_x & valid_y
        
        valid_pts = pts[valid_mask]
        
        if valid_pts.shape[0] == 0:
            return np.zeros_like(self.grid_map, dtype=np.uint8)

        # 3. Apply points to the mask using NumPy indexing (y, x)
        mask = np.zeros_like(self.grid_map, dtype=np.uint8)
        
        # Remember: Map indexing is typically [row (y), col (x)]
        y_indices = valid_pts[:, 1]
        x_indices = valid_pts[:, 0]
        
        mask[y_indices, x_indices] = 1
        
        return mask
    
    def get_front_clearance_ds(self):
        d = self.get_distances()
        if len(d) < 4:
            return float('inf')
        return float(min(d[0], d[2]))  # fl, fr

    def get_back_clearance_ds(self):
        d = self.get_distances()
        if len(d) < 4:
            return float('inf')
        return float(min(d[1], d[3]))  # rl, rr
    


    def align_to_green_carpet(self):
        """
        New alignment strategy:
        - Move backward until green color disappears from the bottom of the image.
        - This ensures the robot is at the edge of the green carpet, not on top of it.
        """
        MAX_STEPS = 300
        steps = 0
        BOTTOM_STRIP_HEIGHT = 30  # pixels from bottom to monitor
        MIN_GREEN_PIXELS = 50     # threshold to consider green "present"

        # Start moving backward
        back_speed = -3.0  # negative for backward
        self.set_robot_velocity(back_speed, back_speed)

        while self.step(self.time_step) != -1:
            steps += 1
            if steps > MAX_STEPS:
                self.stop_motor()
                print("Green alignment timeout (max steps reached).")
                break

            hsv_img = self.get_hsv_image()
            if hsv_img is None:
                self.stop_motor()
                print("Camera error during alignment.")
                break

            height, width, _ = hsv_img.shape

            # Extract bottom strip of image
            bottom_strip = hsv_img[height - BOTTOM_STRIP_HEIGHT:, :]

            # Detect green in the bottom strip using centralized CONSTANTS
            green_mask = utils.segment_color(bottom_strip, 'green')

            green_pixels = cv2.countNonZero(green_mask)

            # Check if green has disappeared from the bottom
            if green_pixels < MIN_GREEN_PIXELS:
                self.stop_motor()
                break

            # Continue backward
            self.set_robot_velocity(back_speed, back_speed)

    def mark_green_carpet_permanently(self, min_pixel_threshold=10, new_area_threshold=0.5):
        """
        Checks if the detected green carpet is close enough and is either a new area
        or a significant expansion of a previously marked area, then aligns and marks it permanently.

        Args:
            min_pixel_threshold (int): Minimum number of pixels required to consider the carpet 'close'.
            new_area_threshold (float): Minimum proportion of points that must be unmarked for the area to be considered 'new'.
        
        Returns:
            bool: True if a permanent mark was successfully made, False otherwise.
        """
        self.green_carpet_active = True
        
        # --- STEP 0: QUICK DETECTION (PRE-ALIGNMENT) ---
        quick_map_points = self.get_green_carpet_points()
        quick_detection_size = quick_map_points.shape[0]

        if quick_detection_size < min_pixel_threshold:
            self.green_carpet_active = False
            return False

        # --- Quick Data Preparation ---
        pts = quick_map_points.astype(np.int32)
        y_indices = pts[:, 1]
        x_indices = pts[:, 0]
        H, W = self.grid_map.shape
        valid = (x_indices >= 0) & (x_indices < W) & (y_indices >= 0) & (y_indices < H)
        x_indices = x_indices[valid]
        y_indices = y_indices[valid]

        if x_indices.size == 0:
            self.green_carpet_active = False
            return False

        # --- STEP 1: NEWNESS CHECK (BEFORE ALIGNMENT) ---
        try:
            green_protect_mask = (self.grid_map == GREEN_CARPET)
        except NameError:
            green_protect_mask = np.zeros_like(self.grid_map, dtype=bool)

        newly_detected_cells = ~green_protect_mask[y_indices, x_indices]
        num_new_cells = np.sum(newly_detected_cells)
        newness_ratio = num_new_cells / x_indices.size
        
        if newness_ratio < new_area_threshold:
            self.green_carpet_active = False
            return False
        
        # --- STEP 2: PROXIMITY & EXPANSION CHECK ---
        current_centroid = np.array([np.mean(x_indices), np.mean(y_indices)])
        closest_patch_index = -1
        min_distance = float('inf')
        
        for i, (old_x, old_y, old_size) in enumerate(self.green_carpet_patches):
            old_centroid = np.array([old_x, old_y])
            distance_to_old_mark = np.linalg.norm(current_centroid - old_centroid)
            
            if distance_to_old_mark < min_distance:
                min_distance = distance_to_old_mark
                closest_patch_index = i
        
        if closest_patch_index != -1 and min_distance < self.green_carpet_proximity_threshold:
            # Current detection is close to a known patch (Geographical Cooldown)
            old_size = self.green_carpet_patches[closest_patch_index][2]
            current_detection_size = int(x_indices.size)
            
            # Apply Expansion Rule: Only proceed if this is a significant size increase
            if current_detection_size > old_size * 1.2: 
                # EXPANSION: Update the record for the old patch with the new, larger size
                self.green_carpet_patches[closest_patch_index] = (current_centroid[0], current_centroid[1], current_detection_size)
            else:
                # Too close and not large enough: Skip marking
                self.green_carpet_active = False
                return False 
            
        for _ in range(15):
            if self.step(self.time_step) == -1:
                break

        
        # --- STEP 3: ALIGNMENT (CALIBRATION) - ONLY IF WORTH MARKING ---
        self.align_to_green_carpet()
        self.stop_motor()
        time.sleep(1)
        
        # --- STEP 4: FULL DETECTION (POST-ALIGNMENT) ---
        current_map_points = self.get_green_carpet_points() 
        current_detection_size = current_map_points.shape[0]

        if current_detection_size < min_pixel_threshold:
            self.green_carpet_active = False
            return False

        # --- Full Data Preparation ---
        pts = current_map_points.astype(np.int32)
        y_indices = pts[:, 1]
        x_indices = pts[:, 0]
        H, W = self.grid_map.shape
        valid = (x_indices >= 0) & (x_indices < W) & (y_indices >= 0) & (y_indices < H)
        x_indices = x_indices[valid]
        y_indices = y_indices[valid]

        if x_indices.size == 0:
            self.green_carpet_active = False
            return False

        # --- STEP 5: Final Permanent Mark ---
        # Grid map update with validated points
        try:
            self.grid_map[y_indices, x_indices] = int(GREEN_CARPET)
            
            # If this was a genuinely NEW patch, add it to the list.
            if min_distance >= self.green_carpet_proximity_threshold:
                self.green_carpet_patches.append((current_centroid[0], current_centroid[1], current_detection_size))
            
            self.last_green_mark_time = time.time()
            self.last_green_carpet_points = [(int(pts[i, 0]), int(pts[i, 1])) for i in range(len(pts))]

            self.green_carpet_active = False
            return True
            
        except Exception as e:
            print(f"[warning] Failed to apply permanent green mark: {e}")
            self.green_carpet_active = False
            return False
            
    def convert_to_map_coordinate_matrix(self, points_world):
        # Compute transformation from world to map:
        # - Scaling (1 / RESOLUTION)
        # - Translation to shift origin to center of map

        # Rotation matrix (identity — no rotation needed in this case)
        R_map = np.array([
            [1 / RESOLUTION, 0],
            [0, -1 / RESOLUTION]  # Flip y-axis
        ])

        # Translation: move origin to center of map
        t_map = np.array([MAP_SIZE // 2, MAP_SIZE // 2])

        # Apply matrix transformation
        points_scaled = points_world @ R_map.T
        points_map = points_scaled + t_map

        return points_map.astype(np.int32)
    
    def robot_on_ground(self, max_tan_pitch=0.08):
        """
        Returns True if robot pitch is small enough to safely use LiDAR for mapping.
        max_tan_pitch ≈ tan(max_allowed_pitch_angle)
        """
        try:
            orientation = self.getSelf().getOrientation()

            # Pitch (rotation around Y axis)
            pitch = np.arctan2(-orientation[6], orientation[8])

            # Use tan(pitch) as geometric validity criterion
            return abs(np.tan(pitch)) < max_tan_pitch

        except Exception:
            return False
            


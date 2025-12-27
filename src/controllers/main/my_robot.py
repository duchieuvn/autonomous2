from controller import Supervisor
import numpy as np
from setup import setup_robot
import cv2
import random
import os
import sys
from vis import MapVisualizer
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

    def camera_detection_loop(self):
        """Continuous detection loop with shared variable communication."""
        # Wait for sensors to initialize
        time.sleep(1.0)
        
        while self.camera_thread_running:
            try:
                # Check if camera is ready
                if self.camera_rgb is None:
                    time.sleep(0.5)
                    continue
                    
                # Only write to shared variable if it's None (main thread has processed previous signal)
                with self.detection_lock:
                    if self.camera_detection_signal is None:
                        # Check for red wall
                        red_wall = self.there_is_red_wall()
                        if red_wall:
                            self.camera_detection_signal = 'red_wall'
                            continue

                        # Check for columns
                        color = self.detect_column()
                        if color:
                            update_count = self.blue_pos_update_count if color == 'blue' else self.yellow_pos_update_count
                            self.camera_detection_signal = ('column', color)
                            column_mask = utils.segment_color(self.get_hsv_image(), color)
                            column_distance = self.estimate_column_distance(color) 
                            
                            if self.column_close(column_mask, column_distance):
                                self.center_column_in_view(color)
                                print("Column is close", column_distance)
                                self.mark_column(color)

                            # Gate: only execute this block max 3 times per color
                            elif update_count < 2:
                                self.center_column_in_view(color)
                                column_distance = self.estimate_column_distance(color) 
                                if column_distance is not None:
                                    column_position = self.position_ahead(column_distance / 100) 
                                    column_map_position = self.convert_to_map_coordinates(column_position[0], column_position[1])
                                    self.update_column_estimation(color, column_map_position)
                                    
                                    if color == 'blue':
                                        self.map_object.update_map_point(column_map_position, value=BLUE_COLUMN)
                                        self.blue_pos_update_count += 1
                                    elif color == 'yellow':
                                        self.map_object.update_map_point(column_map_position, value=YELLOW_COLUMN)
                                        self.yellow_pos_update_count += 1
                                
                time.sleep(0.5)  # Check every 500ms

            except Exception as e:
                print(f"[Detection] Error in detection loop: {e}")
                time.sleep(1.0)

    def lidar_update_loop(self):
        """Continuous lidar mapping loop."""
        # Wait for sensors to initialize
        time.sleep(1.0)
        
        while self.lidar_thread_running:
            try:
                # Check if lidar is ready
                if self.lidar is None:
                    time.sleep(0.5)
                    continue
                # DO NOT UPDATE MAP IF ROBOT IS NOT ON GROUND
                if not self.robot_on_ground():
                    time.sleep(0.1)
                    continue
                    
                # Only update map when robot is not turning
                if not self.is_turning():
                    with self.lidar_lock:
                        self.stop_motor()
                        self.step(self.time_step)
                        time.sleep(0.1)  # Allow robot to stabilize
                        self.lidar_update_map()
                time.sleep(0.2)  # Update at same frequency as before

            except Exception as e:
                print(f"[Lidar] Error in lidar update loop: {e}")
                time.sleep(1.0)

    def column_close(self, column_mask, column_distance):
        nonzero_pixels = np.count_nonzero(column_mask)
        ratio = nonzero_pixels / (column_mask.shape[0] * column_mask.shape[1])
        print("----Column ratio:", ratio)
        # If the column occupies more than 40% of the image -> close
        if ratio > 0.34:
            return True
        else:
            return False

    def is_turning(self):
        # Check if the robot is turning, with 10-step cooldown after turning stops
        left_speed = self.motors['fl'].getVelocity()
        right_speed = self.motors['fr'].getVelocity()
        turning_now = abs(left_speed - right_speed) > 0.08
        return turning_now
        
        if turning_now:
            # Currently turning
            self.is_currently_turning = True
            self.steps_since_turning = 0
            return True
        else:
            # Not currently turning
            if self.is_currently_turning:
                # Turning just stopped, start cooldown
                self.steps_since_turning += 1
                if self.steps_since_turning < 3:
                    return True  # Still in cooldown period
                else:
                    # Cooldown complete
                    self.is_currently_turning = False
                    return False
            else:
                # Not turning and not in cooldown
                return False

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
    
    def obstacle_in_front(self):
        distances = self.get_distances()
        if min(distances[0], distances[2]) < 0.3:
            return True
        return False

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

        if np.mean(distances) < 0.3:
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
    
    def follow_local_target(self, map_target):
        # Return True if the robot reached the target
        if self.get_map_distance(map_target) < PATH_FOLLOWING_TARGET_REACH_DISTANCE:
            # self.stop_motor()
            return True
        
        world_target = self.convert_to_world_coordinates(map_target[0], map_target[1])
        v, w = self.dwa_planner(world_target)
        left_speed, right_speed = self.velocity_to_wheel_speeds(v, w)
        self.set_robot_velocity(left_speed, right_speed)
        return False
    
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

        while self.step() != -1:
            current_heading = self.get_heading('rad')
            angle_diff = abs(utils.get_angle_diff(current_heading, initial_heading))

            if angle_diff - rad_to_turn < TURN_ANGLE_COMPLETION_THRESHOLD:  # Allow a small threshold to avoid overshooting
                break

        self.stop_motor()

    def mark_on_map(self, distance_cm, color='blue'):
        # Use provided distance if available; otherwise attempt to estimate.
        if distance_cm is None:
            distance_cm = self.estimate_column_distance(color)

        # If we still couldn't estimate the distance, skip marking.
        if distance_cm is None:
            print(f"[info] No {color} column distance available; skipping mark_on_map")
            return

        if distance_cm < COLOR_DETECTION_DEPTH_THRESHOLD:
            print(f"column {color} pinned at distance {distance_cm} cm")
            if color == 'blue':
                self.start_point = tuple(self.get_map_position()) # Blue
            elif color == 'yellow':
                self.end_point = tuple(self.get_map_position())  # Yellow

    def align_to_red_wall(self):
        print("Aligning to red wall by moving backward...")

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
                print("Alignment timeout reached.")
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
                    print("Alignment complete.")
                    self.stop_motor()
                    break

                # PD control for turning
                turn_speed = Kp * error + Kd * (error - last_error)
                last_error = error
                self.set_robot_velocity(backward_speed + turn_speed, backward_speed - turn_speed)
                self.step(self.time_step)
            else: # No red detected
                self.stop_motor()
                break

    def mark_closure_block(self):
        # After aligning/stopping, mark the corridor/gap in front as closed
        now = time.time()
        if now - getattr(self, 'last_closure_time', 0.0) >= getattr(self, 'closure_cooldown', 5.0):
            try:
                # rectangle parameters (meters)
                forward_m = CLOSURE_MARK_FORWARD
                back_m = CLOSURE_MARK_BACKWARD
                width_m = CLOSURE_MARK_WIDTH
                # mark closure on the grid_map
                self.map_object.mark_closure_rect_simple(forward_m=forward_m, back_m=back_m, width_m=width_m)
                self.last_closure_time = now
                print('[info] Marked corridor closure (rect) on grid_map')
            except Exception as e:
                print(f'[warning] Failed to mark corridor closure: {e}')
        else:
            print('[info] Skipping closure mark due to cooldown')


    def estimate_column_distance(self, color):
        """Estimate horizontal distance to detected column using depth and known height."""
        self.stop_motor()
        
        # Get sensor data
        depth_img = self.get_camera_depth_cm()
        hsv_img = self.get_hsv_image()
        
        if depth_img is None or hsv_img is None:
            print(f"[Error] Failed to get sensor data for {color} column detection.")
            return None
        
        # Segment column in image
        column_mask = utils.segment_color(hsv_img, color)
        if not np.any(column_mask):
            return None
        
        # Extract valid depth values (filter out -1 sentinel)
        depth_values = depth_img[column_mask != 0]
        valid_depths = depth_values[depth_values > 0]
        
        if len(valid_depths) == 0:
            return None
        
        # Calculate horizontal distance using Pythagorean theorem
        max_depth_cm = float(np.max(valid_depths))
        column_height_cm = 125.0
        
        if max_depth_cm <= column_height_cm:
            return None
        
        horizontal_distance_cm = np.sqrt(max_depth_cm ** 2 - column_height_cm ** 2)
        return horizontal_distance_cm 

    def align_to_column(self, color):
        print("Aligning to column...")

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
                    print("Alignment complete.")
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
                print("Centering timeout reached.")
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

    def handle_frontier_exploration(self, count, map_diff, vis):
        frontier_regions = []
        chosen_frontier = None
        path_to_frontier = None

        if count >= EXPLORATION_START_FRONTIER_AFTER and count % EXPLORATION_FRONTIER_SELECTION_FREQ == 0:
            frontier_regions = self.map_object.detect_frontiers()

            # Occasionally bias frontier choice near known column estimates/start/end
            if random.random() < 0.6:
                print("---Near column frontier")
                chosen_frontier = self.select_frontier_near_known_points(frontier_regions)

            # Fallback to existing selection logic if none chosen
            if chosen_frontier is None:
                if self.chosen_frontier_count < EXPLORATION_MAP_UPDATE_FREQ:
                    print("---Nearest frontier")
                    chosen_frontier = self.select_frontier_target(frontier_regions)
                    self.chosen_frontier_count += 1
                else:
                    print("---Random frontier")
                    chosen_frontier = self.select_frontier_target2(frontier_regions)
                    self.chosen_frontier_count = 0

            if chosen_frontier:
                path_to_frontier = self.map_object.find_path_for_frontier(self.get_map_position(), chosen_frontier)
                if path_to_frontier:
                    self.frontier_following(path_to_frontier, vis)

        return frontier_regions, chosen_frontier, path_to_frontier
    
    def mark_column(self, color):
        if color == 'blue':
            self.start_point = self.get_map_position() 
        elif color == 'yellow':
            self.end_point = self.get_map_position()


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

    def explore(self, debug=True):
        '''
        1. Find blue
        2. Find yellow
        3. Explore randomly, occasionally follow frontier targets

        TODO:
        - improve column distance estimation (far or close)
        - Stop when both columns found
        - Path following from start to end in another function
        '''

        # Start continuous detection and lidar threads before any setup
        self.start_camera_thread()
        self.start_lidar_thread()

        map_object = self.map_object
        vis = None
        if debug:
            vis = MapVisualizer()

        count = 0
        chosen_frontier = None  # store the currently selected frontier
        # planning results/state holders - initialize to avoid stale values between iterations
        path_to_frontier = None
        previous_map = map_object.grid_map.copy()
        map_diff = 1.0
        frontier_regions = []
        last_position = self.get_position()

        while self.step(self.time_step) != -1 and not self.found_all_2_columns():
            if debug:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()

            # Check for camera detection signals from background thread
            with self.detection_lock:
                if self.camera_detection_signal is not None:
                    signal = self.camera_detection_signal
                    self.camera_detection_signal = None  # Reset signal to allow camera thread to set new ones

                    if signal == 'red_wall':
                        self.mark_closure_block()
                        self.align_to_red_wall()
                        # print('Done align to red wall')
                        random_duration = random.randint(700, 900)
                        self.turn_right_milisecond(random_duration)

                    # elif signal[0] == 'column':
                    #     _, color = signal
                    #     column_distance = self.estimate_column_distance(color) 
                    #     if column_distance is not None:
                    #         column_position = self.position_ahead(column_distance / 100) 
                    #         column_map_position = self.convert_to_map_coordinates(column_position[0], column_position[1])
                    #         self.update_column_position(color, column_map_position)
                            
                    #         if color == 'blue':
                    #             map_object.update_map_point(column_map_position, value=BLUE_COLUMN) 
                    #         elif color == 'yellow':
                    #             map_object.update_map_point(column_map_position, value=YELLOW_COLUMN)
                            
                        # self.align_to_column(color)
                        # self.mark_on_map(column_distance, color)

            # --- Random exploration movement ---
            if map_diff < 0.02:
                self.adapt_direction()
                self.set_robot_velocity(MOTOR_VELOCITY_FORWARD, MOTOR_VELOCITY_FORWARD)
                
            # if count % 30 == 0:
            #     if self.robot_stuck(last_position, stuck_distance=0.16):
            #         self.recover_from_stuck(turn_duration=(700, 900))
            #     last_position = self.get_position()

            # --- Periodic green-carpet marking (low-frequency, with cooldown) ---
            if count % 20 == 0:
                try:
                    # Use the larger min_pixel_threshold to ensure robot is close enough
                    marked = self.mark_green_carpet_permanently(min_pixel_threshold=7000)
                    if marked:
                        # If a mark was made, update map with lidar points to protect new area
                        try:
                            points = self.get_pointcloud_world_coordinates()
                            map_points = self.convert_to_map_coordinate_matrix(points)
                            # If GridMap API exists, call map update helpers
                            if hasattr(self, 'bresenham_to_obstacle_score'):
                                self.bresenham_to_obstacle_score(map_points)
                            if hasattr(self.map_object, 'update_grid_map'):
                                self.map_object.update_grid_map()
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[Green Carpet] marking attempt failed: {e}")

            map_diff = utils.percentage_map_differences(previous_map, map_object.grid_map)
            frontier_regions, chosen_frontier, path_to_frontier = self.handle_frontier_exploration(count, map_diff, vis)

            previous_map = map_object.grid_map.copy()

            if debug:
                display_map = map_object.grid_map.copy()
                frontier_overlay_points = []

                # Color frontier regions based on their size
                if len(frontier_regions):
                    region_sizes = [len(region) for region in frontier_regions]

                    if region_sizes:
                        min_size = min(region_sizes)
                        max_size = max(region_sizes)
                        size_range = max_size - min_size if max_size > min_size else 1

                        frontier_color_lookup = {
                            FRONTIER_VISUALIZATION_COLOR_SMALL: (0, 0, 100),      # blue
                            FRONTIER_VISUALIZATION_COLOR_MEDIUM: (0, 200, 155),    # cyan
                            FRONTIER_VISUALIZATION_COLOR_LARGE: (100, 100, 0),     # yellow
                            FRONTIER_VISUALIZATION_COLOR_LARGEST: (100, 0, 0),     # red
                        }

                        for region in frontier_regions:
                            region_size = len(region)
                            size_normalized = (region_size - min_size) / size_range if size_range > 0 else 0

                            if size_normalized < 0.33:
                                color_value = FRONTIER_VISUALIZATION_COLOR_SMALL
                            elif size_normalized < 0.66:
                                color_value = FRONTIER_VISUALIZATION_COLOR_MEDIUM
                            elif size_normalized < 0.95:
                                color_value = FRONTIER_VISUALIZATION_COLOR_LARGE
                            else:
                                color_value = FRONTIER_VISUALIZATION_COLOR_LARGEST

                            rgb = frontier_color_lookup.get(color_value, (255, 0, 0))

                            # Apply color to all cells in the region
                            for x, y in region:
                                if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                                    display_map[y, x] = color_value
                                    frontier_overlay_points.append((x, y, rgb))

                vis.display(display_map)

                # Draw frontier overlays with explicit colors to avoid color-map conflicts
                for fx, fy, fcolor in frontier_overlay_points:
                    vis.draw_point(fx, fy, color=fcolor, radius=2)

                if path_to_frontier:
                    vis.draw_path(path_to_frontier)

                rx, ry = self.get_map_position()
                vis.draw_point(rx, ry, color=(0, 0, 255), radius=5)
                if chosen_frontier:
                    vis.draw_point(chosen_frontier[0], chosen_frontier[1], color=(255, 0, 0), radius=5)
                
                # Draw start and end points if found
                if self.blue_estimated_pos is not None:
                    vis.draw_point(int(self.blue_estimated_pos[0]), int(self.blue_estimated_pos[1]), color=(0, 255, 255), radius=7)
                if self.yellow_estimated_pos is not None:
                    vis.draw_point(int(self.yellow_estimated_pos[0]), int(self.yellow_estimated_pos[1]), color=(255, 255, 0), radius=7)
                
                pygame.display.flip()

            count += 1

        # Clean up detection and lidar threads
        self.stop_camera_thread()
        self.stop_lidar_thread()
        self.stop_motor()
        return self.path


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

    def frontier_following(self, path, vis=None, replan_interval=40):
        """
        Frontier following with:
        - timestep-based replanning
        - front obstacle recovery (reverse → wait → replan)
        - safe fallback behavior
        """

        if path is None or len(path) == 0:
            return

        frontier_goal = path[-1]          # Fixed goal
        current_path = list(path)
        timestep_counter = 0
        stuck_counter = 0

        target_index = 5
        OBSTACLE_THRESHOLD = 0.22   # meters
        REVERSE_DISTANCE = 0.12     # meters
        MAP_UPDATE_WAIT_STEPS = 6   # allow LiDAR/map to update

        while target_index < len(current_path):
            target = current_path[target_index]

            while self.step(self.time_step) != -1 and self.camera_detection_signal is None:
                timestep_counter += 1
                last_position = self.get_position()

                # --------------------------------------------------
                # 🚧 FRONT OBSTACLE → REVERSE → WAIT → REPLAN
                # --------------------------------------------------
                front_dist = self.get_min_front_distance()

                if front_dist < OBSTACLE_THRESHOLD:
                    print(f"[Frontier] Obstacle ahead at {front_dist:.2f} m")

                    # Stop immediately
                    self.stop_motor()

                    # --- Small reverse ---
                    back_speed = 0.12  # m/s
                    dt = TIME_STEP / 1000.0
                    steps = max(1, int(REVERSE_DISTANCE / (back_speed * dt)))

                    lw, rw = self.velocity_to_wheel_speeds(-back_speed, 0.0)
                    self.set_robot_velocity(lw, rw)

                    for _ in range(steps):
                        if self.step(self.time_step) == -1:
                            break

                    self.stop_motor()

                    # --- IMPORTANT: wait for map update ---
                    for _ in range(MAP_UPDATE_WAIT_STEPS):
                        if self.step(self.time_step) == -1:
                            break

                    # --- Replan using updated map ---
                    try:
                        current_start = self.get_map_position()
                        new_path = self.map_object.find_path_for_frontier(
                            current_start,
                            frontier_goal
                        )

                        if new_path and len(new_path) > 5:
                            print("[Frontier] Replanned after obstacle")
                            current_path = new_path
                            target_index = 5
                            break
                        else:
                            print("[Frontier] Replan failed, dropping frontier")
                            return
                    except Exception as e:
                        print(f"[Frontier] Replan error: {e}")
                        return

                # --------------------------------------------------
                # 🔁 PERIODIC TIMESTEP-BASED REPLANNING
                # --------------------------------------------------
                if timestep_counter % replan_interval == 0:
                    try:
                        current_start = self.get_map_position()
                        new_path = self.map_object.find_path_for_frontier(
                            current_start,
                            frontier_goal
                        )

                        if new_path and len(new_path) > 5:
                            print("[Replan] Periodic path update")
                            current_path = new_path
                            target_index = 5
                            break
                    except Exception as e:
                        print(f"[Replan] Failed: {e}")

                # --------------------------------------------------
                # Visualization (unchanged)
                # --------------------------------------------------
                if vis:
                    vis.display(self.map_object.grid_map)
                    vis.draw_path(current_path)
                    rx, ry = self.get_map_position()
                    vis.draw_point(rx, ry, color=(0, 0, 255), radius=5)
                    vis.draw_point(frontier_goal[0], frontier_goal[1], color=(255, 0, 0), radius=5)
                    pygame.display.flip()

                # --------------------------------------------------
                # Follow local target
                # --------------------------------------------------
                if self.follow_local_target(target):
                    stuck_counter = 0
                    break

                # --------------------------------------------------
                # Original stuck logic (unchanged)
                # --------------------------------------------------
                if self.robot_stuck(last_position, stuck_distance=0.08):
                    stuck_counter += 1
                    if stuck_counter > 50:
                        print("Stuck in frontier_following, drop path...")
                        self.stop_motor()
                        self.recover_from_stuck()
                        return

            target_index += 5

        self.stop_motor()




    def path_following_pipeline(self, path, vis=None, frontier_target=None):
        """Follow a path. If a Visualizer `vis` is provided, reuse it to avoid recreating
        the Pygame window (which causes flicker). If not provided, create one once.
        
        Shows the path overlaid on the occupancy grid map.
        """
        # if vis is None:
        #     vis = Visualizer()
        
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
        
        path = path[10::4]
        # Try to align to the first path waypoint before moving to avoid overshoot
        if len(path) > 0:
            try:
                self.align_to_path(path[0])
            except Exception:
                pass
        count = 0
        stuck_counter = 0
        last_position = self.get_position()

        # sample the path for follow targets
        sampled_targets = list(path[::10])
        if len(sampled_targets) == 0:
            self.stop_motor()
            return False

        for idx, target in enumerate(sampled_targets):
            stuck_counter = 0  # Reset stuck counter for each target
            while self.step() != -1:
                for event in pygame.event.get(): 
                    if event.type == pygame.QUIT:
                        exit()
                
                # Get front distance to wall for monitoring
                front_distance = self.get_min_front_distance()

                if vis:
                    vis.display(self.map_object.grid_map, path)
                    if frontier_target:
                        vis.draw_point(frontier_target[0], frontier_target[1], color=(0, 200, 255), radius=5)
                    
                    rx, ry = self.get_map_position()
                    vis.draw_point(rx, ry, color=(0, 0, 255), radius=5)
                    vis.draw_point(target[0], target[1], color=(0, 255, 0), radius=3)
                    pygame.display.flip()
                
                # --- Check for wall blocking the path before stuck detection ---
                wall_threshold = 0.2  # meters (20cm) - wall very close
                if front_distance < wall_threshold:
                    print(f'[warning] Wall detected at {front_distance:.2f}m blocking path to target {target}. Skipping frontier.')
                    if frontier_target is not None:
                        self.visited_frontiers.append(frontier_target)
                    self.stop_motor()
                    return False
                
                # --- Simple stuck detection: if no progress in 50 steps, try recovery ---
                cur_pos = self.get_position()
                dist_moved = float(np.linalg.norm(cur_pos - last_position))
                
                if dist_moved < 0.1:  # Less than 1cm movement
                    stuck_counter += 1
                else:
                    stuck_counter = 0  # Reset if we moved
                    last_position = cur_pos
                
                if stuck_counter > 50:
                    print(f'[info] Path following stuck for 50 steps, recovering')
                    self.recover_from_stuck()
                    stuck_counter = 0
                    last_position = self.get_position()

                reached_target = self.follow_local_target(target)
                if reached_target:
                    print(f'Reached target: {target} (front distance to wall: {front_distance:.2f}m)')
                    # If this is the last sampled target, we've reached the frontier
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
        """
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
        self.set_robot_velocity(-8, -8)
        self.step(300)
        print('Recover from stuck')

        random_duration = random.randint(turn_duration[0], turn_duration[1])
        self.turn_right_milisecond(random_duration)
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

    def select_frontier_near_known_points(self, frontier_regions, top_k=3, max_jitter=5):
        """Bias frontier selection toward regions near known columns/start/end.

        - Uses estimated column positions (blue/yellow) and start/end points if available.
        - Takes up to `top_k` closest frontier regions to any known point, then chooses a random
          cell from one of those regions with a small jitter.
        - Returns None if no anchors or no frontier regions exist.
        """
        if not frontier_regions:
            return None

        anchors = [p for p in [self.start_point, self.end_point, self.blue_estimated_pos, self.yellow_estimated_pos] if p is not None]
        if len(anchors) == 0:
            return None

        region_distances = []
        for region in frontier_regions:
            region_cells = np.array(region)
            centroid = np.mean(region_cells, axis=0)
            dists = [np.linalg.norm(centroid - np.array(a)) for a in anchors]
            region_distances.append((min(dists), region))

        region_distances.sort(key=lambda x: x[0])
        candidate_regions = [r for _, r in region_distances[:max(1, top_k)]]

        chosen_region = random.choice(candidate_regions)
        region_cells = np.array(chosen_region)
        cell = random.choice(region_cells)
        jitter_x = random.randint(-max_jitter, max_jitter)
        jitter_y = random.randint(-max_jitter, max_jitter)
        return (int(cell[0] + jitter_x), int(cell[1] + jitter_y))


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
        # **ACTION REQUIRED: TUNE THESE VALUES**
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        
        green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
        
        h_start = int(self.cam_height * 0.5) 
        green_mask[:h_start, :] = 0
        
        if cv2.countNonZero(green_mask) < 50:
            return np.array([])

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
    

    def align_to_green_carpet(self):
            """
            Aligns the robot to center the detected green carpet in the camera view.
            Uses a P-controller based on the centroid of the green mask.
            """
            print("Aligning to green carpet...")

            # P controller constants
            Kp = 0.015  # Proportional gain for turning (Increased gain for faster reaction)
            error_threshold = 10  # Pixel tolerance for centering
            
            # Moderate forward speed to maintain momentum while aligning
            forward_speed = 3.0  

            while self.step(self.time_step) != -1:
                hsv_img = self.get_hsv_image()
                if hsv_img is None:
                    self.stop_motor()
                    break

                height, width, _ = hsv_img.shape
                
                # Use the bottom half image for reliable ground-level detection
                h_start = int(height * 0.5) 
                hsv_cropped = hsv_img[h_start:, :]

                # Segment green color (using the same tuned values as get_green_carpet_points)
                lower_green = np.array([40, 40, 40])
                upper_green = np.array([80, 255, 255])
                green_mask = cv2.inRange(hsv_cropped, lower_green, upper_green)
                
                M = cv2.moments(green_mask)
                
                if M["m00"] > 0:
                    # Calculate centroid of the green mask (cX is pixel column)
                    cX = int(M["m10"] / M["m00"])
                    # Adjust cX by the cropped height offset to center against the full width
                    # error relative to the center of the *original* camera frame width
                    error = cX - (width // 2)

                    # Check for completion
                    if abs(error) < error_threshold:
                        print("Green carpet alignment complete.")
                        self.stop_motor()
                        break

                    # P-control for turning
                    turn_speed = Kp * error
                    
                    # Apply differential speed (forward_speed is common)
                    self.set_robot_velocity(forward_speed + turn_speed, forward_speed - turn_speed)
                    self.step(self.time_step)
                else: 
                    # Green carpet lost or not visible enough to align
                    self.stop_motor()
                    print("Green carpet lost during alignment.")
                    break

    def mark_green_carpet_permanently(self, min_pixel_threshold=1000, new_area_threshold=0.5):
        """
        Checks if the detected green carpet is close enough and is either a new area
        or a significant expansion of a previously marked area, then aligns and marks it permanently.

        Args:
            min_pixel_threshold (int): Minimum number of pixels required to consider the carpet 'close'.
            new_area_threshold (float): Minimum proportion of points that must be unmarked for the area to be considered 'new'.
        
        Returns:
            bool: True if a permanent mark was successfully made, False otherwise.
        """
        
        # 1. Detect Green Carpet Points & Proximity Check
        current_map_points = self.get_green_carpet_points() 
        current_detection_size = current_map_points.shape[0]

        if current_detection_size < min_pixel_threshold:
            # Carpet is either not detected or not close enough (size filter)
            return False

        # --- Data Preparation ---
        pts = current_map_points.astype(np.int32)
        y_indices = pts[:, 1]
        x_indices = pts[:, 0]
        
        # Calculate Current Centroid (Mean map coordinates)
        current_centroid = np.array([np.mean(x_indices), np.mean(y_indices)])

        # --- 2. Multiple Patch Cooldown Check & Expansion Rule ---
        
        # Default status: Assume it's a new, unknown patch
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
            
            # Apply Expansion Rule: Only proceed if this is a significant size increase (e.g., 20% larger)
            if current_detection_size > old_size * 1.2: 
                # EXPANSION: Update the record for the old patch with the new, larger size
                self.green_carpet_patches[closest_patch_index] = (current_centroid[0], current_centroid[1], current_detection_size)
                print(f"[Green Carpet] Expanding patch at [{old_x:.0f} {old_y:.0f}]. New size: {current_detection_size}")
                # We still need to proceed to Newness check and Marking to physically update the map.
            else:
                # Too close and not large enough: Skip marking to prevent shrinking/jitter noise.
                # print(f"[Green Carpet] Skipping mark. Too close to known patch at [{old_x:.0f} {old_y:.0f}].")
                return False 
        
        # --- 3. Newness Check (Runs only if it's a new patch or passed Expansion Rule) ---
        
        try:
            green_protect_mask = (self.grid_map == GREEN_CARPET)
        except NameError:
            green_protect_mask = np.zeros_like(self.grid_map, dtype=bool)

        newly_detected_cells = ~green_protect_mask[y_indices, x_indices]
        num_new_cells = np.sum(newly_detected_cells)
        newness_ratio = num_new_cells / current_detection_size
        
        if newness_ratio < new_area_threshold:
            # Area is already mostly marked (This is the final guardrail against unnecessary marking)
            return False

        # --- 4. Align and Final Permanent Mark ---
        
        print(f"[Green Carpet] NEW area detected ({current_detection_size} pixels) and close enough. Aligning...")
        
        # Align the robot to center the patch
        self.align_to_green_carpet() 
        self.stop_motor()

        # Re-run detection after alignment for the final map mark
        final_map_points = self.get_green_carpet_points()
        
        if final_map_points.shape[0] > 0:
            final_pts = final_map_points.astype(np.int32)
            final_y = final_pts[:, 1]
            final_x = final_pts[:, 0]
            final_size = final_map_points.shape[0]
            
            # --- FINAL PERMANENT MARKING ---
            try:
                self.grid_map[final_y, final_x] = int(GREEN_CARPET)
                
                # If this was a genuinely NEW patch, add it to the list.
                # If it was an expansion, the list item was already updated in Step 2.
                if min_distance >= self.green_carpet_proximity_threshold:
                    self.green_carpet_patches.append((current_centroid[0], current_centroid[1], final_size))
                
                self.last_green_mark_time = time.time()
                self.last_green_carpet_points = [tuple(p) for p in final_pts]

                print(f"[Map Update] Successfully marked new persistent green carpet ({final_size} pixels).")
                return True
            except Exception as e:
                print(f"[warning] Failed to apply permanent green mark: {e}")
                return False
        else:
            print("[Green Carpet] Lost detection after alignment.")
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
            

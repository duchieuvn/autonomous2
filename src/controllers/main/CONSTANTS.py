TIME_STEP = 32
MAX_VELOCITY = 36
WHEEL_RADIUS = 0.043
AXLE_LENGTH = 0.18
MAP_SIZE = 300 # 10m x 10m grid map
# 300 pixels = 10m, so each pixel is 0.0333m = 3.33cm
RESOLUTION = 10.0 / MAP_SIZE

INITIAL_LOG_ODD = 0
OBSTACLE = 1
FREESPACE = 0
UNKNOWN = 255
BLUE_COLUMN = 100
YELLOW_COLUMN = 150

# Special marker value used to mark temporary closures created by the robot
# Must be different from FREESPACE_VALUE (0) and UNKNOWN_VALUE (255)
CLOSED = 200
# Special marker value for green carpet detection
GREEN_CARPET = 180
# Minimum acceptable path length (meters) for final start->end planning.
# If a found path is shorter than this, the planner will retry with larger
# obstacle inflation to avoid tiny-gap shortcuts.
PATH_MIN_LENGTH_M = 0.8


# ============================================================================
# HYPERPARAMETERS - Centralized configuration for easy tuning
# ============================================================================

# --- EXPLORATION & TIMING ---
EXPLORATION_STEP_STUCK_CHECK = 15          # [USED] Check if robot is stuck every N steps
EXPLORATION_MAP_UPDATE_FREQ = 20           # [USED] Update map every N steps
EXPLORATION_FRONTIER_SELECTION_FREQ = 50   # [USED] Select new frontier target every N steps
EXPLORATION_START_FRONTIER_AFTER = 50  # [USED] Start frontier selection after N exploration steps
EXPLORATION_PATH_PLANNING_FREQ = 100       # [USED] Plan global path every N steps

# --- FRONTIER DETECTION & TARGETING ---
FRONTIER_MIN_DISTANCE_NEW = 20              # [USED] Minimum distance (pixels) to consider frontier as new
FRONTIER_APPROACH_DISTANCE = 10             # [USED] Distance (pixels) to consider frontier reached
FRONTIER_VISUALIZATION_COLOR_SMALL = 50     # [USED] Color value for small frontier regions (blue)
FRONTIER_VISUALIZATION_COLOR_MEDIUM = 100   # [USED] Color value for medium frontier regions (cyan)
FRONTIER_VISUALIZATION_COLOR_LARGE = 200    # [USED] Color value for large frontier regions (yellow)
FRONTIER_VISUALIZATION_COLOR_LARGEST = 220  # [USED] Color value for largest frontier regions (red)

# --- WALL & OBSTACLE DETECTION ---
WALL_DETECTION_THRESHOLD_FRONTIER = 0.3     # [USED] Distance (meters) to consider wall blocking frontier
WALL_DETECTION_THRESHOLD_PATH_FOLLOWING = 0.2  # [HARDCODED] Distance (meters) to consider wall blocking path (used as hardcoded 0.2 in path_following_pipeline)
OBSTACLE_AVOID_THRESHOLD = 0.25             # [USED] Distance (meters) for obstacle avoidance during exploration
OBSTACLE_AVOID_MAX_ATTEMPTS = 3             # [USED] Max attempts to avoid obstacle before giving up

# --- DWA PLANNER (Dynamic Window Approach) ---
DWA_VELOCITY_SAMPLES = [0.1, 0.15, 0.2, 0.25, 0.4, 0.45]  # [USED] Velocity samples (m/s)
DWA_ANGULAR_SAMPLES = [0, 2, 2.5, -2, -2.5, 3, -3, 3.5, -3.5, 4.5, -4.5]  # [USED] Angular velocity samples
DWA_HEADING_WEIGHT = 4.0                    # [USED] Weight for heading score in DWA
DWA_DISTANCE_WEIGHT = 3.5                   # [USED] Weight for distance score in DWA
DWA_SPEED_WEIGHT = 0.05                     # [USED] Weight for speed score in DWA
DWA_PREDICTION_DISTANCE_THRESHOLD = 0.2     # [USED] Max allowed distance increase during prediction

# --- PATH FOLLOWING ---
PATH_FOLLOWING_WAYPOINT_SAMPLE = 10         # [UNUSED] Sample path every N points for waypoints (hardcoded [::10] in path_following_pipeline)
PATH_FOLLOWING_TARGET_REACH_DISTANCE = 4    # [USED] Distance (pixels) to consider waypoint reached
PATH_FOLLOWING_STUCK_THRESHOLD_STEPS = 50  # [UNUSED] Consider stuck after N steps (hardcoded 50 in path_following_pipeline)
PATH_FOLLOWING_STUCK_THRESHOLD_DIST = 0.01  # [UNUSED] Minimum movement (meters) to reset stuck counter (hardcoded 0.01 in path_following_pipeline)

# --- MOTION CONTROL ---
MOTOR_VELOCITY_FORWARD = 8                 # [USED] Forward velocity during random exploration
MOTOR_VELOCITY_TURN = 8                     # [HARDCODED] Turning velocity (used as hardcoded 8 in turn methods)
MOTOR_VELOCITY_BACKWARD = -8                # [HARDCODED] Backward velocity (used as hardcoded -8 in recovery)
RED_WALL_ALIGNMENT_SPEED = -2.0             # [USED] Speed (m/s) for red wall alignment

# --- PID/ALIGNMENT CONTROL ---
ALIGN_RED_WALL_KP = 0.008                   # [USED] Proportional gain for red wall alignment
ALIGN_RED_WALL_KD = 0.002                   # [USED] Derivative gain for red wall alignment
ALIGN_RED_WALL_ERROR_THRESHOLD = 5          # [USED] Error threshold for red wall alignment
ALIGN_COLUMN_KP = 0.008                     # [USED] Proportional gain for column alignment
ALIGN_COLUMN_KD = 0.002                     # [USED] Derivative gain for column alignment
ALIGN_COLUMN_ERROR_THRESHOLD = 20           # [USED] Error threshold for column alignment
ALIGN_COLUMN_FORWARD_SPEED = 2.0            # [USED] Forward speed during column alignment
ALIGN_PATH_ANGLE_THRESHOLD = 15             # [UNUSED] Angle threshold (degrees) for path alignment (defined but not used in align_to_path method)
ALIGN_PATH_CLEAR_DISTANCE = 0.6             # [UNUSED] Required clear distance for rotation without backup (defined but not used)
ALIGN_PATH_BACK_DISTANCE = 0.18             # [UNUSED] Backup distance if not enough clear space (defined but not used)
ALIGN_PATH_ROTATION_SPEED = 1.0             # [UNUSED] Angular speed (rad/s) for path alignment (defined but not used)

# --- CLOSURE MARKING ---
CLOSURE_MARK_COOLDOWN = 5.0                 # [UNUSED] Cooldown (seconds) between closure markings (used as hardcoded in mark_closure_rect_simple)
CLOSURE_MARK_FORWARD = 0.7                  # [USED] Forward distance (meters) for closure rectangle
CLOSURE_MARK_BACKWARD = 0.05                # [USED] Backward distance (meters) for closure rectangle
CLOSURE_MARK_WIDTH = 0.6                    # [USED] Width (meters) of closure rectangle
CLOSURE_MARK_IOU_THRESHOLD = 0.4            # [UNUSED] IoU threshold to skip re-marking same closure (hardcoded 0.4 in mark_closure_rect_simple)

# --- COLOR DETECTION ---
COLOR_DETECTION_DEPTH_THRESHOLD = 80        # [USED] Depth threshold (cm) for column detection
COLOR_DETECTION_RED_PIXEL_RATIO = 0.25      # [USED] Pixel ratio threshold for red wall detection
RED_WALL_HSV_LOWER1 = [0, 120, 70]    # [USED] Red color lower bound 1 (HSV)
RED_WALL_HSV_UPPER1 = [10, 255, 255]  # [USED] Red color upper bound 1 (HSV)
RED_WALL_HSV_LOWER2 = [170, 120, 70]  # [USED] Red color lower bound 2 (HSV)
RED_WALL_HSV_UPPER2 = [180, 255, 255] # [USED] Red color upper bound 2 (HSV)

# --- TURNING & TURNING COOLDOWN ---
TURN_ROTATION_COOLDOWN_STEPS = 10           # [UNUSED] Steps cooldown after turning stops (used as hardcoded 10 in is_turning method)
TURN_DURATION_MIN = 50                      # [USED] Minimum turn duration (milliseconds)
TURN_DURATION_MAX = 200                     # [USED] Maximum turn duration (milliseconds)
TURN_ANGLE_COMPLETION_THRESHOLD = 0.05      # [USED] Angle threshold (radians) to complete turn

# --- SENSOR & DISTANCE ---
LIDAR_FRONT_CONE_ANGLE = 15                 # [USED] Angle (degrees) for front distance detection cone
LIDAR_FALLBACK_FRONT_DISTANCE = float('inf') # [UNUSED] Fallback distance if lidar unavailable (not referenced in code)
TURNING_THRESHOLD = 0.01                    # [UNUSED] Speed difference to detect turning (not referenced in code, similar logic hardcoded as 0.01)

# --- MAPPING & GRID ---
MAPPING_LOG_ODDS_FREE = -0.36               # [UNUSED] Log-odds value for free space (hardcoded 0.85 in bresenham_to_obstacle_score)
MAPPING_LOG_ODDS_OCCUPIED = 0.85            # [UNUSED] Log-odds value for occupied space (hardcoded 0.85 in bresenham_to_obstacle_score)
MAPPING_LOG_ODDS_CLIP_MIN = -5              # [UNUSED] Minimum log-odds clipping value (hardcoded -5 in update_grid_map)
MAPPING_LOG_ODDS_CLIP_MAX = 5               # [UNUSED] Maximum log-odds clipping value (hardcoded 5 in update_grid_map)
MAPPING_PROBABILITY_OBSTACLE = 0.7          # [UNUSED] Probability threshold for obstacle (hardcoded 0.7 in update_grid_map)
MAPPING_PROBABILITY_FREE = 0.5              # [UNUSED] Probability threshold for free space (hardcoded 0.5 in update_grid_map)

# --- A* PATHFINDING ---
ASTAR_INFLATION_LEVELS = [2.5, 3, 3.5, 4]   # [USED] Inflation levels for escalating A*
ASTAR_EXPANSION_PIXELS = 3                 # [USED] Expansion around start/end points
ASTAR_FRONTIER_INFLATION = 4              # [UNUSED] Inflation level for frontier pathfinding (hardcoded 13 in find_path_for_frontier)

# ============================================================================
# END HYPERPARAMETERS
# ============================================================================
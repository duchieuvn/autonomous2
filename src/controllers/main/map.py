import utils
from CONSTANTS import *
import numpy as np
import cv2
from collections import deque
from astar_clearance import runAStarSearch  # NEW: Clearance-aware A* with tunable hyperparameters
from astar_2_spline import runAStarSearch as runAStarSearchSpline
from astar_standard import runAStarSearch as runAStarSearch2Spline
import random 

class GridMap():
    """Encapsulates grid map management including occupancy grid, log-odds mapping, frontier detection, and pathfinding."""
    
    def __init__(self, robot=None):
        """Initialize the GridMap.
        
        Args:
            robot: Reference to MyRobot instance for coordinate conversions and sensor access.
        """
        self.robot = robot
        self.map_size = MAP_SIZE
        self.resolution = RESOLUTION  # meters per pixel    
        self.log_odds = np.full((self.map_size, self.map_size), INITIAL_LOG_ODD, dtype=np.float32)
        self.grid_map = np.full((self.map_size, self.map_size), UNKNOWN, dtype=np.uint8)
        self.frontier_regions = []
        self.visited_frontiers = []

    def there_is_obstacle(self, map_target):
        """Check if a map target contains an obstacle."""
        if self.grid_map[map_target[1], map_target[0]] == OBSTACLE:
            return True
        return False

    def update_log_odds(self, robot_pos, lidar_map_points):
        """Update log-odds map using Bresenham for each LIDAR point in map coordinates."""
            
        for map_target in lidar_map_points:
            points = utils.bresenham_line(robot_pos, map_target)

            # Free points: all except the last
            for x, y in points[:-1]:
                if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                    self.log_odds[y, x] -= 0.36

            # Occupied cell: last one
            x, y = points[-1]
            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                self.log_odds[y, x] += 0.85

    def update_grid_map(self):
        """Update grid_map from log-odds obstacle_score_map.
        
        Protects special marker cells (CLOSED, GREEN_CARPET, START, END) from being overwritten.
        Converts log-odds probabilities to discrete map values:
        - P > 0.7: OBSTACLE (1) = white
        - P < 0.5: FREESPACE (0) = black
        - Otherwise: UNKNOWN (255) = gray (unchanged)
        """
        # Clip the score map from (-5, 5) to avoid overflow when applying exponential function
        limited_score_map = np.clip(self.log_odds, -5, 5)
        # Higher score -> Higher P closer to 1
        P = 1 / (1 + np.exp(-limited_score_map))
        

        # Build protection masks for cells that should not be overwritten by sensor updates
        closed_mask = (self.grid_map == CLOSED)
        green_protect_mask = (self.grid_map == GREEN_CARPET)

        # Combine all protection masks
        protected_mask = closed_mask | green_protect_mask 

        # Only update unprotected cells based on probability thresholds
        unknown_mask = (self.log_odds == INITIAL_LOG_ODD)
        obstacle_mask = (P > 0.7) & (~protected_mask)
        free_mask = (P < 0.5) & (~protected_mask)

        # 1. Update obstacles and free space
        self.grid_map[obstacle_mask] = OBSTACLE
        self.grid_map[free_mask] = FREESPACE
        
        # 2. Restore all protected cells at the end
        self.grid_map[green_protect_mask] = GREEN_CARPET
        # self.grid_map[closed_mask] = CLOSED

        # Update obstacles and free space
        self.grid_map[obstacle_mask] = OBSTACLE
        self.grid_map[free_mask] = FREESPACE
        self.grid_map[unknown_mask] = UNKNOWN

    def lidar_update_grid_map(self, robot_pos, lidar_points):
        map_points = self.convert_to_map_coordinate_matrix(lidar_points)
        self.update_log_odds(robot_pos, map_points)
        self.update_grid_map()

    def update_map_point(self, map_point, value):
        """Update a single point in grid_map with the specified value."""
        x, y = map_point
        if 0 <= x < self.map_size and 0 <= y < self.map_size:
            self.grid_map[y, x] = value

    def mark_closure_rect_simple(self, forward_m=0.7, back_m=-0.2, width_m=0.6, value=CLOSED):
        """Mark a simple rectangular closure directly in front of the robot on grid_map.

        Rectangle is defined in robot-local frame: x forward, y lateral.
        The rectangle spans from -back_m (behind robot) to +forward_m in front.
        """
        if self.robot is None:
            return False
            
        # Skip grid updates while robot is turning to avoid noise corruption
        if self.robot.is_turning():
            return False
        
        # Robot world position and heading
        rx, ry = self.robot.get_position()
        heading = self.robot.get_heading('rad')

        # Rectangle corners in local robot frame
        hw = width_m / 2.0
        front_offset = getattr(self.robot, 'axle_length', 0.0) / 2.0
        front_x = front_offset + forward_m
        rear_x = front_offset - back_m
        corners_local = np.array([
            [front_x, hw],
            [front_x, -hw],
            [rear_x, -hw],
            [rear_x, hw]
        ])

        # Rotation matrix
        R = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
        # Transform corners to world
        corners_world = corners_local @ R.T + np.array([rx, ry])

        # Convert to map coordinates
        map_pts = [self.robot.convert_to_map_coordinates(float(x), float(y)) for (x, y) in corners_world]
        pts = np.array(map_pts, dtype=np.int32).reshape((-1, 1, 2))

        # Build mask for bounding box
        try:
            mask = np.zeros_like(self.grid_map, dtype=np.uint8)
            cv2.fillPoly(mask, [pts], color=1)
            poly_area = int(mask.sum())
            if poly_area == 0:
                return False

            existing_closed = (self.grid_map == value).astype(np.uint8)
            overlap = int((existing_closed & mask).sum())
            overlap_ratio = overlap / poly_area
            IOU_SKIP_THRESHOLD = 0.4
            if overlap_ratio >= IOU_SKIP_THRESHOLD:
                return False

            ys, xs = np.where(mask)
            x0, x1 = max(0, xs.min()), min(self.grid_map.shape[1] - 1, xs.max())
            y0, y1 = max(0, ys.min()), min(self.grid_map.shape[0] - 1, ys.max())

            roi = self.grid_map[y0:y1+1, x0:x1+1]
            roi_mask = mask[y0:y1+1, x0:x1+1]
            roi[roi_mask == 1] = int(value)
            self.grid_map[y0:y1+1, x0:x1+1] = roi
            return True
        except Exception as e:
            print(f'[warning] mark_closure_rect_simple failed: {e}')
            return False

    def detect_frontiers(self):
        """Detects frontier cells (free space next to unknown space) and groups them into regions using BFS clustering."""
        frontier_cells = []
        map_size = self.map_size
        
        # Define 4-connected neighbors for frontier detection
        NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for x in range(1, map_size - 1):
            for y in range(1, map_size - 1):
                # Check if cell is Freespace
                if self.grid_map[y, x] == FREESPACE:
                    # Check 4-connected neighbors for Unknown space
                    if any(
                        self.grid_map[y + dy, x + dx] == UNKNOWN
                        for dx, dy in NEIGHBORS_4
                    ):
                        frontier_cells.append((x, y))
        
        # Group frontier cells into regions using BFS clustering
        frontier_regions = self._cluster_frontiers_bfs(frontier_cells)
        self.frontier_regions = frontier_regions
        return frontier_regions
        
    def _cluster_frontiers_bfs(self, frontier_cells, min_cluster_size=15):
        """Groups frontier cells into regions using Breadth-First Search (BFS) 
        and 8-connected neighbors. Filters out small clusters.
        """
        if not frontier_cells:
            print("No frontier cells found.")
            return []

        frontier_set = set(frontier_cells)
        visited = set()
        clusters = []
        map_size = self.map_size
        NEIGHBORS_8 = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        # BFS to find connected components
        for start_cell in frontier_cells:
            if start_cell not in visited:
                cluster = []
                queue = deque([start_cell])
                visited.add(start_cell)

                while queue:
                    x, y = queue.popleft()
                    cluster.append((x, y))

                    for dx, dy in NEIGHBORS_8:
                        nx, ny = x + dx, y + dy
                        
                        if (0 < nx < map_size - 1 and 
                            0 < ny < map_size - 1 and
                            (nx, ny) in frontier_set and
                            (nx, ny) not in visited):
                            
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                
                # Filter small clusters
                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)

        return clusters



    def find_path(self, start_point, end_point):
        """Plan a path from start to end using clearance-aware A* with escalating inflation."""
        inflation_attempts = ASTAR_INFLATION_LEVELS
        best_path = None
        best_len = 0.0

        for inflation_pixels in inflation_attempts:
            global_map = self.grid_map.copy().astype(np.float32)
            global_map = utils.remove_noisy_pixels(global_map, obstacle_value=OBSTACLE, connectivity=4)
            global_map = utils.inflate_obstacles(global_map, inflation_pixels=inflation_pixels)
            utils.expand_free_pixel(global_map, end_point, inflation_pixels=ASTAR_EXPANSION_PIXELS)
            utils.expand_free_pixel(global_map, start_point, inflation_pixels=ASTAR_EXPANSION_PIXELS)

            path = runAStarSearchSpline(global_map, start_point, end_point)

            if path is None or len(path) <= 1:
                continue

            # Compute total path length in meters
            try:
                total_len = 0.0
                prev_w = self.robot.convert_to_world_coordinates(path[0][0], path[0][1])
                for p in path[1:]:
                    cur_w = self.robot.convert_to_world_coordinates(p[0], p[1])
                    dx = cur_w[0] - prev_w[0]
                    dy = cur_w[1] - prev_w[1]
                    total_len += (dx*dx + dy*dy) ** 0.5
                    prev_w = cur_w
            except Exception as e:
                print(f"[warning] find_path length computation failed: {e}")
                total_len = 0.0

            if total_len >= PATH_MIN_LENGTH_M:
                if inflation_pixels != inflation_attempts[0]:
                    print(f"[info] find_path: selected path with inflation={inflation_pixels} px length={total_len:.2f}m")
                return path

            if total_len > best_len:
                best_len = total_len
                best_path = path

        if best_path is not None:
            print(f"[info] find_path: no path met minimum {PATH_MIN_LENGTH_M}m; returning longest candidate {best_len:.2f}m")
            return best_path

        # Fallback: try planning on raw grid
        try:
            raw_map = self.grid_map.copy().astype(np.float32)
            raw_map = utils.remove_noisy_pixels(raw_map, obstacle_value=OBSTACLE, connectivity=4)
            raw_path = runAStarSearchSpline(raw_map, start_point, end_point)
            if raw_path is not None and len(raw_path) > 1:
                try:
                    raw_len = 0.0
                    prev_w = self.robot.convert_to_world_coordinates(raw_path[0][0], raw_path[0][1])
                    for p in raw_path[1:]:
                        cur_w = self.robot.convert_to_world_coordinates(p[0], p[1])
                        dx = cur_w[0] - prev_w[0]
                        dy = cur_w[1] - prev_w[1]
                        raw_len += (dx*dx + dy*dy) ** 0.5
                        prev_w = cur_w
                except Exception:
                    raw_len = 0.0
                print(f"[info] find_path: no inflated candidate long enough; returning raw-map path length={raw_len:.2f}m")
                return raw_path
        except Exception as e:
            print(f"[warning] find_path raw-map fallback failed: {e}")

        print(f"[info] find_path: no path found between points after inflation attempts; returning None")
        return None

    def find_path_for_frontier(self, start_point, end_point):
        if start_point is None or end_point is None:
            return []
        
        """Frontier-specific path finder using clearance-aware A*."""
        global_map = self.grid_map.copy().astype(np.float32)
        closed_mask = (global_map == CLOSED)

        # Temporary map: treat closed cells as free to avoid inflation expansion
        temp_map = global_map.copy()
        temp_map[closed_mask] = FREESPACE

        # Run standard preprocessing on temp_map
        temp_map = utils.inflate_obstacles(temp_map, inflation_pixels=ASTAR_FRONTIER_INFLATION)

        # Re-apply closed pixels as hard obstacles
        temp_map[closed_mask] = OBSTACLE

        global_map = temp_map
        utils.expand_free_pixel(global_map, end_point, inflation_pixels=ASTAR_EXPANSION_PIXELS)
        utils.expand_free_pixel(global_map, start_point, inflation_pixels=ASTAR_EXPANSION_PIXELS)
        cv2.imwrite("debug_frontier_map.png", global_map*255)
        path = runAStarSearchSpline(global_map, start_point, end_point)
        return path


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
  
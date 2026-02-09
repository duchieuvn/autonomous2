import utils
from CONSTANTS import *
import numpy as np
import cv2
from collections import deque
from astar_2_spline import runAStarSearch as runAStarSearchSpline
import random
import pygame
import threading 

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
        
        # Visualization state
        self.current_path = None
        self.robot_position = None
        self.target_position = None
        self.column_points = []
        self.pygame_screen = None
        self.window_size = (800, 800)
        self.should_stop_visualization = False
        self.visualization_thread = None
        self.vis_lock = threading.Lock()
        self.clock = None
        self.font = None
        self.color_map = {
            FREESPACE: (255, 255, 255),   # freespace: white
            OBSTACLE: (0, 0, 0),         # obstacle: black
            UNKNOWN: (80, 80, 80),    # unknown: gray
            BLUE_COLUMN: (0, 0, 255),     # start: blue
            YELLOW_COLUMN: (255, 255, 0),     # end: yellow
            180: (0, 255, 255),   # frontier generic: cyan
            50: (0, 0, 255),      # small frontier: blue
            101: (0, 255, 255),   # medium frontier: cyan
            200: (255, 255, 0),   # large frontier: yellow
            220: (255, 0, 0),     # largest frontier: red
            CLOSED: (128, 0, 128),  # closed area: purple
        }

    def there_is_obstacle(self, map_target):
        """Check if a map target contains an obstacle."""
        cell = self.grid_map[map_target[1], map_target[0]]

        if cell == OBSTACLE or cell == GREEN_CARPET or cell == CLOSED:
            return True

        return False

    def update_log_odds(self, robot_pos, lidar_map_points):
        """Update log-odds map using Bresenham for each LIDAR point in map coordinates.
        
        Only updates cells with log_odds < 3. Cells with log_odds >= 3 are locked and remain unchanged.
        """
            
        for map_target in lidar_map_points:
            points = utils.bresenham_line(robot_pos, map_target)

            # Free points: all except the last
            for x, y in points[:-1]:
                if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                    if self.log_odds[y, x] < 3.5:
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

        protected_mask = closed_mask | green_protect_mask

        # Only update unprotected cells based on probability thresholds
        unknown_mask = (self.log_odds == INITIAL_LOG_ODD) & (~protected_mask)
        obstacle_mask = (P > 0.7) & (~protected_mask)
        free_mask = (P < 0.5) & (~protected_mask)

        # Apply updates
        self.grid_map[obstacle_mask] = OBSTACLE
        self.grid_map[free_mask] = FREESPACE
        self.grid_map[unknown_mask] = UNKNOWN

        # Restore protected cells (guarantee persistence)
        self.grid_map[green_protect_mask] = GREEN_CARPET
        self.grid_map[closed_mask] = CLOSED



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
            closed_mask = (global_map == CLOSED)
            green_mask  = (global_map == GREEN_CARPET)

            # --- do NOT inflate closures/green: make them free during inflation ---
            temp_map = global_map.copy()
            temp_map[closed_mask] = FREESPACE
            temp_map[green_mask]  = OBSTACLE

            # preprocessing + inflate
            temp_map = utils.clean_small_obstacle_components(temp_map, obstacle_value=OBSTACLE, min_size=6, connectivity=4)
            temp_map = utils.remove_noisy_pixels(temp_map, obstacle_value=OBSTACLE, connectivity=4)
            temp_map = utils.inflate_obstacles(temp_map, inflation_pixels=inflation_pixels)

            # --- re-apply closures/green as hard obstacles after inflation ---
            temp_map[closed_mask] = OBSTACLE
            temp_map[green_mask]  = OBSTACLE

            global_map = temp_map

            utils.expand_free_pixel(global_map, end_point, inflation_pixels=ASTAR_EXPANSION_PIXELS)
            utils.expand_free_pixel(global_map, start_point, inflation_pixels=ASTAR_EXPANSION_PIXELS)
            # debugging output
            cv2.imwrite(f"debug_inflated_map_{inflation_pixels}px.png", global_map*255)

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
        # try:
        #     raw_map = self.grid_map.copy().astype(np.float32)
        #     raw_map = utils.remove_noisy_pixels(raw_map, obstacle_value=OBSTACLE, connectivity=4)
        #     utils.expand_free_pixel(raw_map, end_point, inflation_pixels=ASTAR_EXPANSION_PIXELS)
        #     utils.expand_free_pixel(raw_map, start_point, inflation_pixels=ASTAR_EXPANSION_PIXELS)
        #     # debugging output
        #     cv2.imwrite("debug_raw_map.png", raw_map*255)

        #     raw_path = runAStarSearchSpline(raw_map, start_point, end_point)
        #     if raw_path is not None and len(raw_path) > 1:
        #         try:
        #             raw_len = 0.0
        #             prev_w = self.robot.convert_to_world_coordinates(raw_path[0][0], raw_path[0][1])
        #             for p in raw_path[1:]:
        #                 cur_w = self.robot.convert_to_world_coordinates(p[0], p[1])
        #                 dx = cur_w[0] - prev_w[0]
        #                 dy = cur_w[1] - prev_w[1]
        #                 raw_len += (dx*dx + dy*dy) ** 0.5
        #                 prev_w = cur_w
        #         except Exception:
        #             raw_len = 0.0
        #         print(f"[info] find_path: no inflated candidate long enough; returning raw-map path length={raw_len:.2f}m")
        #         return raw_path
        # except Exception as e:
        #     print(f"[warning] find_path raw-map fallback failed: {e}")

        # print(f"[info] find_path: no path found between points after inflation attempts; returning None")
        # return None


    def find_path_for_frontier(self, start_point, end_point):
        if start_point is None or end_point is None:
            return []
        
        """Frontier-specific path finder using clearance-aware A*."""
        global_map = self.grid_map.copy().astype(np.float32)
        global_map = utils.clean_small_obstacle_components(global_map, obstacle_value=OBSTACLE, min_size=6, connectivity=4)
        closed_mask = (global_map == CLOSED)
        green_mask  = (global_map == GREEN_CARPET)

        # Temporary map: treat closed cells as free to avoid inflation expansion
        temp_map = global_map.copy()
        temp_map[closed_mask] = FREESPACE
        temp_map[green_mask]  = OBSTACLE
        

        # Run standard preprocessing on temp_map
        temp_map = utils.inflate_obstacles(temp_map, inflation_pixels=ASTAR_FRONTIER_INFLATION)

        # Re-apply closed pixels as hard obstacles
        temp_map[closed_mask] = OBSTACLE
        temp_map[green_mask]  = OBSTACLE

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

    # ========== VISUALIZATION METHODS ==========
    
    def _init_pygame(self):
        """Initialize pygame display for visualization."""
        pygame.init()
        pygame.font.init()
        self.pygame_screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Grid Map Visualizer - Live")
        self.clock = pygame.time.Clock()
        # A small, readable font for overlay text
        self.font = pygame.font.SysFont("Arial", 16)
    
    def _grid_to_display(self, grid_map):
        """Convert grid map to RGB display image using color_map.
        
        Args:
            grid_map: 2D numpy array with grid values
            
        Returns:
            RGB image (height, width, 3) for display
        """
        h, w = grid_map.shape
        display = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Map each grid value to color
        for value, rgb in self.color_map.items():
            mask = grid_map == value
            display[mask] = rgb
        
        # Handle unmapped values as black
        mapped_mask = np.zeros((h, w), dtype=bool)
        for value in self.color_map.keys():
            mapped_mask |= (grid_map == value)
        display[~mapped_mask] = (0, 0, 0)
        
        return display
    
    def _map_to_screen(self, map_x, map_y):
        """Convert map coordinates to screen coordinates."""
        screen_x = int(map_x * self.window_size[0] / self.map_size)
        screen_y = int(map_y * self.window_size[1] / self.map_size)
        return screen_x, screen_y
    
    def _draw_path(self, path, color=(255, 0, 0), thickness=2):
        """Draw a path on screen.
        
        Args:
            path: List of (x, y) map coordinates
            color: RGB tuple for path color
            thickness: Line thickness in pixels
        """
        if not path or len(path) < 2:
            return
        scaled_path = [self._map_to_screen(x, y) for x, y in path]
        for i in range(len(scaled_path) - 1):
            pygame.draw.line(self.pygame_screen, color, scaled_path[i], scaled_path[i + 1], thickness)
    
    def _draw_point(self, map_x, map_y, color=(0, 255, 0), radius=5):
        """Draw a single point on screen.
        
        Args:
            map_x, map_y: Map coordinates
            color: RGB tuple for point color
            radius: Circle radius in pixels
        """
        if map_x is None or map_y is None:
            return
        screen_x, screen_y = self._map_to_screen(map_x, map_y)
        pygame.draw.circle(self.pygame_screen, color, (screen_x, screen_y), radius)
    
    def _draw_frontier_regions(self, display_map):
        """Draw frontier regions colored by size on the display map.
        
        Modifies display_map in place to color frontier cells.
        """
        if not self.frontier_regions:
            return
        
        # Sort frontier regions by size
        sorted_regions = sorted(self.frontier_regions, key=lambda r: len(r))
        n_regions = len(sorted_regions)
        
        if n_regions == 0:
            return
        
        # Color frontiers by size gradient
        for idx, region in enumerate(sorted_regions):
            if n_regions == 1:
                color_value = 101  # medium cyan
            elif idx == 0:
                color_value = 50   # smallest: blue
            elif idx == n_regions - 1:
                color_value = 220  # largest: red
            elif idx < n_regions / 2:
                color_value = 101  # medium: cyan
            else:
                color_value = 200  # large: yellow
            
            for x, y in region:
                if 0 <= x < self.map_size and 0 <= y < self.map_size:
                    display_map[y, x] = color_value
    
    def _draw_columns(self):
        """Draw detected column points."""
        if not self.column_points:
            return
        
        for col_data in self.column_points:
            if isinstance(col_data, tuple) and len(col_data) >= 3:
                x, y, color = col_data[0], col_data[1], col_data[2]
                self._draw_point(x, y, color=color, radius=5)
            elif isinstance(col_data, (list, tuple)) and len(col_data) >= 2:
                x, y = col_data[0], col_data[1]
                self._draw_point(x, y, color=(0, 255, 255), radius=5)
    
    def run_visualization_loop(self):
        """Main visualization loop that continuously displays map state.
        
        Runs independently in a separate thread, polling GridMap state and
        rendering all elements (grid, frontiers, paths, robot, targets, columns).
        """
        self._init_pygame()
        
        while not self.should_stop_visualization:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.should_stop_visualization = True
                    break
            
            # Acquire lock for thread-safe reading
            with self.vis_lock:
                # Create display map with frontier regions
                display_map = self.grid_map.copy()
                self._draw_frontier_regions(display_map)
                
                # Convert to RGB image
                display_img = self._grid_to_display(display_map)
                
                # Copy path and positions for rendering
                current_path = self.current_path
                robot_pos = self.robot_position
                target_pos = self.target_position
                columns = self.column_points.copy() if self.column_points else []
                start_point = getattr(self.robot, 'start_point', None) if self.robot else None
                end_point = getattr(self.robot, 'end_point', None) if self.robot else None
            
            # Resize and blit to screen
            resized = cv2.resize(display_img, self.window_size, interpolation=cv2.INTER_NEAREST)
            surface = pygame.surfarray.make_surface(np.transpose(resized, (1, 0, 2)))
            self.pygame_screen.blit(surface, (0, 0))
            
            # Draw overlays in order (path -> target -> columns -> robot)
            if current_path:
                self._draw_path(current_path, color=(255, 0, 0), thickness=2)
            
            if target_pos:
                self._draw_point(target_pos[0], target_pos[1], color=(0, 255, 0), radius=4)
            
            # Draw estimated columns with distinct colors and larger radius
            for col_data in columns:
                if isinstance(col_data, tuple) and len(col_data) >= 3:
                    x, y, color = col_data[0], col_data[1], col_data[2]
                    # Draw filled circle for estimated column
                    self._draw_point(x, y, color=color, radius=6)
                    # Draw circle outline for clarity
                    # screen_x, screen_y = self._map_to_screen(x, y)
                    # pygame.draw.circle(self.pygame_screen, (255, 255, 255), (screen_x, screen_y), 8, 2)
            
            # Draw marked start/end points on the map with distinct markers
            if start_point is not None:
                self._draw_point(start_point[0], start_point[1], color=(0, 0, 255), radius=8)
                # Draw an outline circle for clarity
                screen_x, screen_y = self._map_to_screen(start_point[0], start_point[1])
                pygame.draw.circle(self.pygame_screen, (255, 255, 255), (screen_x, screen_y), 10, 2)
            
            if end_point is not None:
                self._draw_point(end_point[0], end_point[1], color=(255, 200, 0), radius=8)
                # Draw an outline circle for clarity
                screen_x, screen_y = self._map_to_screen(end_point[0], end_point[1])
                pygame.draw.circle(self.pygame_screen, (255, 255, 255), (screen_x, screen_y), 10, 2)
            
            # Draw robot last (on top) to ensure visibility
            if robot_pos:
                self._draw_point(robot_pos[0], robot_pos[1], color=(0, 0, 255), radius=6)

            # Overlay start/end coordinates when available
            if self.font:
                overlay_y = 8
                if start_point is not None:
                    start_xy = tuple(int(v) for v in start_point)
                    text_surface = self.font.render(f"Start: {start_xy}", True, (0, 0, 255))
                    self.pygame_screen.blit(text_surface, (8, overlay_y))
                    overlay_y += text_surface.get_height() + 4
                if end_point is not None:
                    end_xy = tuple(int(v) for v in end_point)
                    text_surface = self.font.render(f"End:   {end_xy}", True, (255, 200, 0))
                    self.pygame_screen.blit(text_surface, (8, overlay_y))
            
            # Update display
            pygame.display.flip()
            
            # Limit to 30 FPS
            self.clock.tick(30)
        
        # Cleanup
        pygame.quit()
    
    def start_visualization(self):
        """Start the visualization loop in a separate daemon thread."""
        if self.visualization_thread is None or not self.visualization_thread.is_alive():
            self.should_stop_visualization = False
            self.visualization_thread = threading.Thread(target=self.run_visualization_loop, daemon=True)
            self.visualization_thread.start()
            print("[info] Visualization thread started")
    
    def stop_visualization(self):
        """Stop the visualization loop gracefully."""
        self.should_stop_visualization = True
        if self.visualization_thread and self.visualization_thread.is_alive():
            self.visualization_thread.join(timeout=2.0)
            print("[info] Visualization thread stopped")
  
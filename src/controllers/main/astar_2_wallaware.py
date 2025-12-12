import heapq
import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.ndimage import distance_transform_edt

MAX_COST = 99999

def runAStarSearch(global_map, start_coords, goal_coords, wall_bias_strength=0.5, wall_bias_range=3):
    """
    Enhanced A* with wall distance bias
    
    Parameters:
    - wall_bias_strength: How much to penalize being near walls (0.1-2.0, higher = more avoidance)
    - wall_bias_range: Distance range for wall penalty (pixels, typically 2-5)
    """
    rows, cols = global_map.shape

    def isWithinBounds(x, y):
        return 0 <= y < rows and 0 <= x < cols

    def isNearlySame(p1, p2):
        return abs(p1.x - p2.x) <= 2 and abs(p1.y - p2.y) <= 2

    def getReferenceCost(min_heuristic):
        return min_heuristic * (1.2 + 0.5 * np.exp(-0.05 * min_heuristic))

    # Pre-compute distance transform for wall avoidance
    def computeWallDistanceMap():
        """Compute distance to nearest wall for each free cell"""
        # Create binary map (0 = free, 1 = wall)
        binary_map = (global_map > 0).astype(np.uint8)
        
        # Compute distance transform (distance to nearest wall)
        distance_map = distance_transform_edt(1 - binary_map)
        return distance_map

    wall_distance_map = computeWallDistanceMap()

    def getWallPenalty(x, y):
        """Calculate smooth penalty based on distance to walls"""
        if not isWithinBounds(x, y):
            return 0
            
        wall_distance = wall_distance_map[y, x]
        
        if wall_distance >= wall_bias_range:
            return 0  # No penalty if far from walls
        
        # Smooth sigmoid-based penalty instead of exponential
        # This creates gentler transitions
        normalized_distance = wall_distance / wall_bias_range
        penalty_factor = 1 / (1 + np.exp(6 * (normalized_distance - 0.5)))
        
        return wall_bias_strength * penalty_factor * 0.5  # Reduced impact

    def getLocalWallDensity(x, y, radius=2):
        """Calculate local wall density for additional bias"""
        if not isWithinBounds(x, y):
            return 0
            
        wall_count = 0
        total_cells = 0
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if isWithinBounds(nx, ny):
                    total_cells += 1
                    if global_map[ny, nx] > 0:  # Wall
                        wall_count += 1
        
        return wall_count / total_cells if total_cells > 0 else 0

    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.pixel_value = global_map[y, x]
            self.g_cost = MAX_COST
            self.f_cost = MAX_COST
            self.parent = None
            
            # Pre-calculate wall penalties
            self.wall_penalty = getWallPenalty(x, y)
            self.local_wall_density = getLocalWallDensity(x, y)

        def isAccessible(self):
            return self.pixel_value == 0

        def getNeighbors(self):
            directions = [[0,2],[0,-2],[2,0],[-2,0],[2,2],[2,-2],[-2,2],[-2,-2]]
            neighbors = []
            for dx, dy in directions:
                nx, ny = self.x + dx, self.y + dy
                if isWithinBounds(nx, ny):
                    neighbors.append(Point(nx, ny))
            return neighbors

        def heuristic(self, goal):
            # Base heuristic
            dist = np.sqrt((self.x - goal.x) ** 2 + (self.y - goal.y) ** 2)
            base_heuristic = dist * 1.2 + 0.01
            
            # Gentle wall avoidance - only add to heuristic, not g-cost
            # This encourages wall avoidance without creating jerky movements
            wall_avoidance_cost = self.wall_penalty * 0.3  # Much reduced impact
            
            return base_heuristic + wall_avoidance_cost
        
        def __lt__(self, other):
            return self.f_cost < other.f_cost

    class AStarHeap:
        def __init__(self, goal_point):
            self.open_heap = []
            self.cost_matrix = np.full(global_map.shape, MAX_COST)
            self.visited_count = 0
            self.goal = goal_point
            self.current_min_heuristic = MAX_COST
            self.entry_count = 0

        def calculateGCost(self, cur_point, target_point):
            # Keep g-cost simple - only base movement cost
            # This prevents jerky movements caused by varying wall penalties
            base_cost = np.sqrt((cur_point.x - target_point.x) ** 2 + (cur_point.y - target_point.y) ** 2)
            
            # Optional: very gentle wall cost (comment out if still too jerky)
            # wall_cost = target_point.wall_penalty * 0.1
            
            return base_cost  # + wall_cost

        def updateCost(self, point):
            heuristic = point.heuristic(self.goal)
            if point.parent is None:
                point.g_cost = 0
                point.f_cost = 0
            else:
                point.g_cost = point.parent.g_cost + self.calculateGCost(point.parent, point)
                point.f_cost = point.g_cost + heuristic

            if point.f_cost < self.cost_matrix[point.y, point.x]:
                if self.cost_matrix[point.y, point.x] == MAX_COST:
                    self.visited_count += 1
                self.cost_matrix[point.y, point.x] = point.f_cost
                self.current_min_heuristic = min(self.current_min_heuristic, heuristic)
                self.entry_count += 1
                heapq.heappush(self.open_heap, (point.f_cost, self.entry_count, point))
                return True
            return False

        def pop(self):
            if not self.open_heap:
                return None
            return heapq.heappop(self.open_heap)[2]

    def tracePath(goal):
        path = []
        current = goal
        while current.parent is not None:
            path.append((current.x, current.y))
            current = current.parent
        path.append((current.x, current.y))
        return path[::-1]

    def postProcessWallAvoidance(path, push_distance=1.5):
        """
        Gentle post-processing with smoothing to avoid jerky adjustments
        """
        if len(path) < 3:
            return path
            
        adjusted_path = []
        
        for i, (x, y) in enumerate(path):
            current_distance = wall_distance_map[y, x]
            
            # Only adjust if very close to walls and not at start/end points
            if current_distance < push_distance and i > 0 and i < len(path) - 1:
                # Look for gradual adjustment direction based on path flow
                prev_point = path[i-1]
                next_point = path[i+1] if i < len(path) - 1 else (x, y)
                
                # Calculate path direction
                path_dir = np.array(next_point) - np.array(prev_point)
                if np.linalg.norm(path_dir) > 0:
                    path_dir = path_dir / np.linalg.norm(path_dir)
                    
                    # Find perpendicular directions for wall avoidance
                    perp_dirs = [np.array([-path_dir[1], path_dir[0]]), 
                                np.array([path_dir[1], -path_dir[0]])]
                    
                    best_x, best_y = x, y
                    best_distance = current_distance
                    
                    # Try gentle moves in perpendicular directions
                    for perp_dir in perp_dirs:
                        for step in [0.5, 1.0]:  # Small steps
                            nx = int(round(x + perp_dir[0] * step))
                            ny = int(round(y + perp_dir[1] * step))
                            
                            if (isWithinBounds(nx, ny) and 
                                global_map[ny, nx] == 0 and
                                wall_distance_map[ny, nx] > best_distance):
                                best_x, best_y = nx, ny
                                best_distance = wall_distance_map[ny, nx]
                                break
                    
                    adjusted_path.append((best_x, best_y))
                else:
                    adjusted_path.append((x, y))
            else:
                adjusted_path.append((x, y))
        
        # Apply smoothing filter to reduce remaining jerkiness
        return smoothPathPoints(adjusted_path)
    
    def smoothPathPoints(path, window_size=3):
        """Apply moving average to smooth path points"""
        if len(path) < window_size:
            return path
            
        smoothed = [path[0]]  # Keep first point
        
        for i in range(1, len(path) - 1):
            # Calculate weighted average of nearby points
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(path), i + window_size // 2 + 1)
            
            avg_x = sum(p[0] for p in path[start_idx:end_idx]) / (end_idx - start_idx)
            avg_y = sum(p[1] for p in path[start_idx:end_idx]) / (end_idx - start_idx)
            
            # Ensure smoothed point is still in free space
            smooth_x, smooth_y = int(round(avg_x)), int(round(avg_y))
            if (isWithinBounds(smooth_x, smooth_y) and 
                global_map[smooth_y, smooth_x] == 0):
                smoothed.append((smooth_x, smooth_y))
            else:
                smoothed.append(path[i])  # Fall back to original
        
        smoothed.append(path[-1])  # Keep last point
        return smoothed

    def smoothPath(path, method='natural_spline', smoothness=0.3):
        # [Previous smoothPath implementation goes here - keeping it the same]
        if len(path) < 2:
            return path
        
        path = np.array(path, dtype=float)
        x, y = path[:, 0], path[:, 1]
        t = np.arange(len(path))
        
        # Use 'natural' boundary condition for smoother curves
        cs_x = CubicSpline(t, x, bc_type='natural')
        cs_y = CubicSpline(t, y, bc_type='natural')
        
        # Higher sampling rate for smoother curves
        samples_per_segment = max(5, int(20 * smoothness))
        t_fine = np.linspace(0, len(path)-1, num=len(path) * samples_per_segment)
        smooth_x = cs_x(t_fine)
        smooth_y = cs_y(t_fine)
        
        smooth_path = [(int(round(x)), int(round(y))) for x, y in zip(smooth_x, smooth_y)]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_smooth_path = []
        for point in smooth_path:
            if point not in seen:
                seen.add(point)
                unique_smooth_path.append(point)
                
        return unique_smooth_path

    start = Point(*start_coords)
    goal = Point(*goal_coords)

    frontier = AStarHeap(goal)
    frontier.updateCost(start)

    current = frontier.pop()
    total_cells = rows * cols

    while current is not None and not isNearlySame(current, goal) and frontier.visited_count < total_cells:
        for neighbor in current.getNeighbors():
            if neighbor.isAccessible():
                neighbor.parent = current
                if frontier.updateCost(neighbor) and \
                   neighbor.heuristic(goal) < getReferenceCost(frontier.current_min_heuristic):
                    continue
        current = frontier.pop()

    if current is None:
        print("Visited cells:", frontier.visited_count)
        print("No path found")
        return []

    path = tracePath(current)
    
    # Apply post-processing to gently push away from walls
    path = postProcessWallAvoidance(path, push_distance=1.5)
    
    # Apply smoothing
    smooth_path = smoothPath(path, method='natural_spline', smoothness=0.3)
    
    return smooth_path


# Usage examples for smooth wall avoidance:

# For very gentle, smooth wall avoidance (recommended):
# path = runAStarSearch(global_map, start, goal, wall_bias_strength=0.2, wall_bias_range=3)

# For slightly more avoidance while staying smooth:
# path = runAStarSearch(global_map, start, goal, wall_bias_strength=0.4, wall_bias_range=4)

# Alternative: Disable A* wall bias and only use post-processing:
# path = runAStarSearch(global_map, start, goal, wall_bias_strength=0.0, wall_bias_range=0)
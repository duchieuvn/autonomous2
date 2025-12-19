import heapq
import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

MAX_COST = 99999

def runAStarSearch(global_map, start_coords, goal_coords, smoothness_method='curvature_penalty'):
    """
    A* with mathematical smoothness models
    
    Methods:
    - 'curvature_penalty': Penalize high curvature during pathfinding
    - 'acceleration_constraint': Limit path acceleration/jerk
    - 'energy_minimization': Post-process with energy minimization
    - 'variational_smoothing': Solve smoothness as optimization problem
    """
    rows, cols = global_map.shape

    def isWithinBounds(x, y):
        return 0 <= y < rows and 0 <= x < cols

    def isNearlySame(p1, p2):
        return abs(p1.x - p2.x) <= 2 and abs(p1.y - p2.y) <= 2

    # Pre-compute distance transform
    binary_map = (global_map > 0).astype(np.uint8)
    wall_distance_map = distance_transform_edt(1 - binary_map)

    def calculateCurvature(p1, p2, p3):
        """Calculate curvature at point p2 given three consecutive points"""
        if p1 is None or p3 is None:
            return 0
        
        # Vector from p1 to p2
        v1 = np.array([p2.x - p1.x, p2.y - p1.y])
        # Vector from p2 to p3  
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        # Avoid division by zero
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            return 0
        
        # Normalize vectors
        v1_norm, v2_norm = v1 / norm_v1, v2 / norm_v2
        
        # Calculate curvature using cross product magnitude
        cross_product = abs(np.cross(v1_norm, v2_norm))
        return cross_product

    def calculateAcceleration(p1, p2, p3):
        """Calculate acceleration magnitude at point p2"""
        if p1 is None or p3 is None:
            return 0
        
        # Second derivative approximation: acceleration ≈ (p3 - 2*p2 + p1)
        accel = np.array([p3.x - 2*p2.x + p1.x, p3.y - 2*p2.y + p1.y])
        return np.linalg.norm(accel)

    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.pixel_value = global_map[y, x]
            self.g_cost = MAX_COST
            self.f_cost = MAX_COST
            self.parent = None
            self.grandparent = None  # For curvature calculation
            
            # Wall distance for gentle bias
            self.wall_distance = wall_distance_map[y, x] if isWithinBounds(x, y) else 0

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
            
            # Gentle wall distance bias (inverse relationship)
            wall_bias = 0.1 / (1 + self.wall_distance) if self.wall_distance > 0 else 0.1
            
            return base_heuristic + wall_bias
        
        def __lt__(self, other):
            return self.f_cost < other.f_cost

    class SmoothAStarHeap:
        def __init__(self, goal_point, smoothness_method):
            self.open_heap = []
            self.cost_matrix = np.full(global_map.shape, MAX_COST)
            self.visited_count = 0
            self.goal = goal_point
            self.entry_count = 0
            self.smoothness_method = smoothness_method

        def calculateGCost(self, cur_point, target_point):
            """Enhanced G-cost with smoothness penalties"""
            # Base movement cost
            base_cost = np.sqrt((cur_point.x - target_point.x) ** 2 + (cur_point.y - target_point.y) ** 2)
            
            smoothness_penalty = 0
            
            if self.smoothness_method == 'curvature_penalty':
                # Add curvature penalty to discourage sharp turns
                if cur_point.parent is not None:
                    curvature = calculateCurvature(cur_point.parent, cur_point, target_point)
                    smoothness_penalty = curvature * 2.0  # Tunable parameter
                    
            elif self.smoothness_method == 'acceleration_constraint':
                # Add acceleration penalty to limit jerk
                if cur_point.grandparent is not None:
                    acceleration = calculateAcceleration(cur_point.grandparent, cur_point, target_point)
                    smoothness_penalty = acceleration * 0.5  # Tunable parameter
            
            # Gentle wall distance penalty
            wall_penalty = 0.1 / (1 + target_point.wall_distance) if target_point.wall_distance > 0 else 0.1
            
            return base_cost + smoothness_penalty + wall_penalty

        def updateCost(self, point):
            heuristic = point.heuristic(self.goal)
            if point.parent is None:
                point.g_cost = 0
                point.f_cost = 0
            else:
                # Set grandparent for acceleration calculations
                point.grandparent = point.parent.parent if point.parent else None
                
                point.g_cost = point.parent.g_cost + self.calculateGCost(point.parent, point)
                point.f_cost = point.g_cost + heuristic

            if point.f_cost < self.cost_matrix[point.y, point.x]:
                if self.cost_matrix[point.y, point.x] == MAX_COST:
                    self.visited_count += 1
                self.cost_matrix[point.y, point.x] = point.f_cost
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

    def energyMinimizationSmoothing(path, alpha=1.0, beta=10.0, wall_weight=0.1):
        """
        Minimize total energy: E = α∑|acceleration|² + β∑|wall_penalty|²
        This creates mathematically smooth paths
        """
        if len(path) < 3:
            return path
        
        path = np.array(path, dtype=float)
        n = len(path)
        
        def energy_function(coords):
            coords = coords.reshape(-1, 2)
            energy = 0
            
            # Acceleration penalty (second derivative)
            for i in range(1, n-1):
                if i > 0 and i < n-1:
                    accel = coords[i+1] - 2*coords[i] + coords[i-1]
                    energy += alpha * np.sum(accel**2)
            
            # Wall distance penalty
            for i, (x, y) in enumerate(coords):
                if 0 <= int(y) < rows and 0 <= int(x) < cols:
                    wall_dist = wall_distance_map[int(y), int(x)]
                    wall_penalty = 1.0 / (1 + wall_dist)
                    energy += beta * wall_weight * wall_penalty
            
            return energy
        
        # Keep start and end points fixed
        initial_coords = path[1:-1].flatten()
        
        # Optimize interior points
        result = minimize(energy_function, initial_coords, method='BFGS')
        
        if result.success:
            optimized_coords = result.x.reshape(-1, 2)
            # Reconstruct full path
            smooth_path = [path[0]]
            smooth_path.extend(optimized_coords)
            smooth_path.append(path[-1])
            
            # Convert to integers and validate
            smooth_path = [(int(round(x)), int(round(y))) for x, y in smooth_path]
            
            # Ensure all points are in free space
            valid_path = []
            for x, y in smooth_path:
                if isWithinBounds(x, y) and global_map[y, x] == 0:
                    valid_path.append((x, y))
                else:
                    # Find nearest valid point
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if isWithinBounds(nx, ny) and global_map[ny, nx] == 0:
                                valid_path.append((nx, ny))
                                break
                        else:
                            continue
                        break
                    else:
                        # If no valid neighbor found, keep original point from path
                        valid_path.append((x, y))
            
            return valid_path
        
        return path.tolist()

    def variationalSmoothing(path, lambda_smooth=1.0, lambda_wall=0.1):
        """
        Solve smoothing as a linear system: minimize ||Ax - b||² + λ||Lx||²
        where L is the discrete Laplacian (second derivative operator)
        """
        if len(path) < 3:
            return path
        
        path = np.array(path, dtype=float)
        n = len(path)
        
        # Create discrete Laplacian matrix (second derivative operator)
        L = np.zeros((n-2, n))
        for i in range(n-2):
            L[i, i] = 1
            L[i, i+1] = -2
            L[i, i+2] = 1
        
        # Identity matrix for data fidelity
        I = np.eye(n)
        
        # Solve for x-coordinates
        A_x = I + lambda_smooth * L.T @ L
        b_x = path[:, 0]
        smooth_x = np.linalg.solve(A_x, b_x)
        
        # Solve for y-coordinates
        A_y = I + lambda_smooth * L.T @ L
        b_y = path[:, 1]
        smooth_y = np.linalg.solve(A_y, b_y)
        
        # Combine results
        smooth_path = [(int(round(x)), int(round(y))) for x, y in zip(smooth_x, smooth_y)]
        
        # Validate and fix invalid points
        valid_path = []
        for i, (x, y) in enumerate(smooth_path):
            if isWithinBounds(x, y) and global_map[y, x] == 0:
                valid_path.append((x, y))
            else:
                # Use original point if smoothed version is invalid
                valid_path.append(tuple(path[i].astype(int)))
        
        return valid_path

    def gaussianSmoothing(path, sigma=1.0, iterations=1):
        """
        Apply Gaussian smoothing with collision checking
        """
        if len(path) < 3:
            return path
        
        path = np.array(path, dtype=float)
        
        for _ in range(iterations):
            smooth_path = path.copy()
            
            for i in range(1, len(path) - 1):
                # Gaussian kernel weights (3-point)
                weights = [0.25, 0.5, 0.25]
                
                smooth_x = weights[0]*path[i-1, 0] + weights[1]*path[i, 0] + weights[2]*path[i+1, 0]
                smooth_y = weights[0]*path[i-1, 1] + weights[1]*path[i, 1] + weights[2]*path[i+1, 1]
                
                # Check if smoothed point is valid
                sx, sy = int(round(smooth_x)), int(round(smooth_y))
                if isWithinBounds(sx, sy) and global_map[sy, sx] == 0:
                    smooth_path[i] = [smooth_x, smooth_y]
            
            path = smooth_path
        
        return [(int(round(x)), int(round(y))) for x, y in path]

    # Run A* with chosen smoothness method
    start = Point(*start_coords)
    goal = Point(*goal_coords)

    if smoothness_method in ['curvature_penalty', 'acceleration_constraint']:
        frontier = SmoothAStarHeap(goal, smoothness_method)
    else:
        frontier = SmoothAStarHeap(goal, 'none')

    frontier.updateCost(start)
    current = frontier.pop()
    total_cells = rows * cols

    while current is not None and not isNearlySame(current, goal) and frontier.visited_count < total_cells:
        for neighbor in current.getNeighbors():
            if neighbor.isAccessible():
                neighbor.parent = current
                if frontier.updateCost(neighbor):
                    continue
        current = frontier.pop()

    if current is None:
        print("No path found")
        return []

    path = tracePath(current)

    # Apply post-processing smoothing based on method
    if smoothness_method == 'energy_minimization':
        smooth_path = energyMinimizationSmoothing(path, alpha=1.0, beta=5.0)
    elif smoothness_method == 'variational_smoothing':
        smooth_path = variationalSmoothing(path, lambda_smooth=2.0)
    elif smoothness_method == 'gaussian_smoothing':
        smooth_path = gaussianSmoothing(path, sigma=1.0, iterations=2)
    else:
        # Default spline smoothing
        smooth_path = defaultSplineSmoothing(path)

    return smooth_path

def defaultSplineSmoothing(path):
    """Default cubic spline smoothing"""
    if len(path) < 2:
        return path
    
    path = np.array(path, dtype=float)
    x, y = path[:, 0], path[:, 1]
    t = np.arange(len(path))
    
    cs_x = CubicSpline(t, x, bc_type='natural')
    cs_y = CubicSpline(t, y, bc_type='natural')
    
    t_fine = np.linspace(0, len(path)-1, num=len(path) * 10)
    smooth_x = cs_x(t_fine)
    smooth_y = cs_y(t_fine)
    
    smooth_path = [(int(round(x)), int(round(y))) for x, y in zip(smooth_x, smooth_y)]
    
    # Remove duplicates
    seen = set()
    unique_path = []
    for point in smooth_path:
        if point not in seen:
            seen.add(point)
            unique_path.append(point)
    
    return unique_path


# Usage examples:

# Method 1: Curvature penalty during A* search
# path = runAStarSearch(map, start, goal, smoothness_method='curvature_penalty')

# Method 2: Acceleration constraint during A* search  
# path = runAStarSearch(map, start, goal, smoothness_method='acceleration_constraint')

# Method 3: Post-process with energy minimization (most mathematically rigorous)
# path = runAStarSearch(map, start, goal, smoothness_method='energy_minimization')

# Method 4: Variational smoothing (fastest mathematical approach)
# path = runAStarSearch(map, start, goal, smoothness_method='variational_smoothing')

# Method 5: Gaussian smoothing (simple and effective)
# path = runAStarSearch(map, start, goal, smoothness_method='gaussian_smoothing')
import heapq
import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev

MAX_COST = 99999

def runAStarSearch(global_map, start_coords, goal_coords):
    rows, cols = global_map.shape

    def isWithinBounds(x, y):
        return 0 <= y < rows and 0 <= x < cols

    def isNearlySame(p1, p2):
        return abs(p1.x - p2.x) <= 2 and abs(p1.y - p2.y) <= 2

    def getReferenceCost(min_heuristic):
        return min_heuristic * (1.2 + 0.5 * np.exp(-0.05 * min_heuristic))

    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.pixel_value = global_map[y, x]
            self.g_cost = MAX_COST
            self.f_cost = MAX_COST
            self.parent = None

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
            dist = np.sqrt((self.x - goal.x) ** 2 + (self.y - goal.y) ** 2)
            return dist * 1.2 + 0.01
        
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
            return np.sqrt((cur_point.x - target_point.x) ** 2 + (cur_point.y - target_point.y) ** 2)

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

    def detectCorners(path, angle_threshold=45):
        """Detect corners in the path based on direction changes"""
        if len(path) < 3:
            return []
        
        corners = []
        for i in range(1, len(path) - 1):
            # Get vectors for incoming and outgoing segments
            v1 = np.array(path[i]) - np.array(path[i-1])
            v2 = np.array(path[i+1]) - np.array(path[i])
            
            # Calculate angle between vectors
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
                angle = np.degrees(np.arccos(cos_angle))
                
                if angle > angle_threshold:
                    corners.append(i)
        
        return corners

    def smoothPath(path, method='natural_spline', smoothness=0.3):
        """
        Multiple smoothing methods with different characteristics
        
        method options:
        - 'natural_spline': Uses natural boundary conditions (smoother curves)
        - 'parametric_spline': Uses parametric B-spline (very smooth)
        - 'corner_aware': Pre-processes corners for extra smoothness
        - 'bezier_like': Creates Bezier-like curves between waypoints
        """
        if len(path) < 2:
            return path
        
        path = np.array(path, dtype=float)
        
        if method == 'natural_spline':
            return _naturalSplineSmooth(path, smoothness)
        elif method == 'parametric_spline':
            return _parametricSplineSmooth(path, smoothness)
        elif method == 'corner_aware':
            return _cornerAwareSmooth(path, smoothness)
        elif method == 'bezier_like':
            return _bezierLikeSmooth(path, smoothness)
        else:
            return _naturalSplineSmooth(path, smoothness)

    def _naturalSplineSmooth(path, smoothness):
        """Natural spline with relaxed boundary conditions"""
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
        
        return _finalizeSmoothedPath(smooth_x, smooth_y)

    def _parametricSplineSmooth(path, smoothness):
        """Parametric B-spline for very smooth curves"""
        if len(path) < 4:  # Need at least 4 points for B-spline
            return _naturalSplineSmooth(path, smoothness)
        
        x, y = path[:, 0], path[:, 1]
        
        # Fit parametric spline
        # Higher smoothing factor = smoother curves
        s_factor = len(path) * (1 - smoothness) * 10  # Adjust smoothing
        tck, u = splprep([x, y], s=s_factor, k=3)  # k=3 for cubic
        
        # Evaluate spline at fine resolution
        samples = len(path) * max(10, int(30 * smoothness))
        u_fine = np.linspace(0, 1, samples)
        smooth_coords = splev(u_fine, tck)
        
        return _finalizeSmoothedPath(smooth_coords[0], smooth_coords[1])

    def _cornerAwareSmooth(path, smoothness):
        """Pre-process sharp corners for extra smoothness"""
        corners = detectCorners(path, angle_threshold=30)  # Detect sharper corners
        
        # Insert intermediate points at corners for smoother transitions
        enhanced_path = []
        for i, point in enumerate(path):
            enhanced_path.append(point)
            
            if i in corners and i > 0 and i < len(path) - 1:
                # Add intermediate points around corners
                prev_point = path[i-1]
                next_point = path[i+1]
                
                # Create smoother transition points
                t1, t2 = 0.3, 0.7  # Adjust these for different smoothness
                inter1 = prev_point + t1 * (point - prev_point)
                inter2 = point + t2 * (next_point - point)
                
                enhanced_path.insert(-1, inter1)  # Insert before current point
                enhanced_path.append(inter2)      # Insert after current point
        
        enhanced_path = np.array(enhanced_path)
        return _parametricSplineSmooth(enhanced_path, smoothness)

    def _bezierLikeSmooth(path, smoothness):
        """Create Bezier-like smooth curves between waypoints"""
        if len(path) < 3:
            return path.tolist()
        
        smooth_points = [path[0]]  # Start with first point
        
        for i in range(1, len(path) - 1):
            prev_point = path[i-1]
            curr_point = path[i]
            next_point = path[i+1]
            
            # Create control points for smooth transitions
            # The smoothness parameter controls how much we deviate from the original path
            control_distance = smoothness * min(
                np.linalg.norm(curr_point - prev_point),
                np.linalg.norm(next_point - curr_point)
            ) * 0.5
            
            # Direction vectors
            dir_in = (curr_point - prev_point) / np.linalg.norm(curr_point - prev_point)
            dir_out = (next_point - curr_point) / np.linalg.norm(next_point - curr_point)
            
            # Smooth direction (average of input and output directions)
            smooth_dir = (dir_in + dir_out) / 2
            smooth_dir = smooth_dir / np.linalg.norm(smooth_dir)
            
            # Create control points
            control1 = curr_point - smooth_dir * control_distance
            control2 = curr_point + smooth_dir * control_distance
            
            # Generate points along the curve
            num_curve_points = max(3, int(10 * smoothness))
            for t in np.linspace(0, 1, num_curve_points):
                if i == 1 and t == 0:
                    continue  # Skip first point to avoid duplicates
                
                # Simple Bezier-like interpolation
                curve_point = (1-t) * control1 + t * control2
                smooth_points.append(curve_point)
        
        smooth_points.append(path[-1])  # Add final point
        
        smooth_coords = np.array(smooth_points)
        return _finalizeSmoothedPath(smooth_coords[:, 0], smooth_coords[:, 1])

    def _finalizeSmoothedPath(smooth_x, smooth_y):
        """Convert smooth coordinates to integer path and remove duplicates"""
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
    
    # Try different smoothing methods:
    # smooth_path = smoothPath(path, method='natural_spline', smoothness=0.9)
    # smooth_path = smoothPath(path, method='parametric_spline', smoothness=0.7)
    smooth_path = smoothPath(path, method='corner_aware', smoothness=1.0)
    # smooth_path = smoothPath(path, method='bezier_like', smoothness=0.8)
    
    return smooth_path
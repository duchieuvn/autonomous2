"""
Optimized A* Pathfinding with Clearance-Based Cost Function.

This implementation includes:
- Clearance checking to prefer wider paths
- Fallback to closest reachable cell if goal is unreachable
- 8-directional movement (orthogonal and diagonal)
- Tunable hyperparameters for path behavior

Reference: https://github.com/Chrisbelefantis/A-Star-Algorithm/blob/master/Astar-Algorithm.py#L246
Adapted for autonomous robot navigation with clearance optimization.
"""

import heapq
import numpy as np
from math import sqrt

# ============================================================================
# HYPERPARAMETERS (Tunable)
# ============================================================================
# Straight movement cost (orthogonal: N, S, E, W)
STRAIGHT_COST = 1.0

# Diagonal movement cost
DIAGONAL_COST = sqrt(2)

# Clearance penalty scaling factor
# Higher = more preference for wide paths, lower = faster paths preferred
CLEARANCE_PENALTY_SCALE = 2.0

# Clearance penalty floor (minimum penalty even with low clearance)
CLEARANCE_PENALTY_FLOOR = 1.0

# Clearance measurement radius (pixels) - how far to check obstacle distance
CLEARANCE_RADIUS = 3

# Heuristic multiplier (1.0 = admissible, >1.0 = faster but may not find optimal path)
HEURISTIC_MULTIPLIER = 1.0

# Obstacle threshold: values >= this are treated as obstacles
OBSTACLE_THRESHOLD = 0.5

# ============================================================================


def get_clearance(grid_map, x, y, radius=CLEARANCE_RADIUS):
    """
    Calculate clearance (distance to nearest obstacle) at a given position.
    
    Clearance is the minimum distance to any obstacle within the specified radius.
    Returns the actual clearance distance or radius if no obstacles found nearby.
    
    Args:
        grid_map: 2D numpy array of map values
        x, y: Position to check clearance at
        radius: Search radius in pixels
    
    Returns:
        float: Minimum distance to obstacle, or radius if none found
    """
    rows, cols = grid_map.shape
    min_clearance = float(radius)
    
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            
            # Check if this cell is an obstacle
            if grid_map[ny, nx] >= OBSTACLE_THRESHOLD:
                distance = sqrt(dx*dx + dy*dy)
                if distance < min_clearance:
                    min_clearance = distance
    
    return min_clearance


def calculate_cost(current, neighbor, grid_map):
    """
    Calculate movement cost with clearance-based penalty.
    
    Cost = base_movement_cost + clearance_penalty
    
    The clearance penalty encourages the path to stay away from obstacles.
    
    Args:
        current: Current position (x, y)
        neighbor: Neighbor position (x, y)
        grid_map: 2D numpy array of map values
    
    Returns:
        float: Total cost of movement
    """
    # Base movement cost
    dx = neighbor[0] - current[0]
    dy = neighbor[1] - current[1]
    
    if abs(dx) + abs(dy) == 1:
        base_cost = STRAIGHT_COST
    else:
        # Diagonal movement
        base_cost = DIAGONAL_COST
    
    # Clearance-based penalty (encourages wider paths)
    clearance = get_clearance(grid_map, neighbor[0], neighbor[1], CLEARANCE_RADIUS)
    
    # Convert clearance to penalty: lower clearance = higher penalty
    # clearance_penalty = CLEARANCE_PENALTY_FLOOR if clearance == 0 else (CLEARANCE_PENALTY_SCALE / clearance)
    # Simpler linear approach: penalty inversely proportional to clearance
    if clearance == 0:
        clearance_penalty = 10.0  # High penalty for being next to obstacles
    else:
        clearance_penalty = max(CLEARANCE_PENALTY_FLOOR, CLEARANCE_PENALTY_SCALE / clearance)
    
    return base_cost + clearance_penalty


def runAStarSearch(grid_map, start_coords, goal_coords):
    """
    Optimized A* pathfinding with clearance-based cost function.
    
    Features:
    - Prefers wider paths (high clearance) through clearance penalty
    - Finds path to closest reachable cell if exact goal is unreachable
    - 8-directional movement (orthogonal + diagonal)
    - Configurable hyperparameters for tuning behavior
    
    Args:
        grid_map: 2D numpy array where values >= OBSTACLE_THRESHOLD are obstacles
        start_coords: (x, y) starting position
        goal_coords: (x, y) goal position
    
    Returns:
        list: Path from start to goal (or closest reachable cell) as list of (x, y) tuples.
              Empty list if no path found (including if start is unreachable).
    """
    rows, cols = grid_map.shape
    
    def is_valid_cell(x, y):
        """Check if a cell is traversable (not an obstacle)"""
        if not (0 <= x < cols and 0 <= y < rows):
            return False
        return grid_map[y, x] < OBSTACLE_THRESHOLD
    
    def heuristic(pos, goal):
        """Euclidean distance heuristic (admissible, optimistic)"""
        dx = pos[0] - goal[0]
        dy = pos[1] - goal[1]
        distance = sqrt(dx*dx + dy*dy)
        return HEURISTIC_MULTIPLIER * distance
    
    def reconstruct_path(came_from, current):
        """Reconstruct path from start to current by backtracking"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
    
    start = tuple(start_coords)
    goal = tuple(goal_coords)
    
    # Validate start position
    if not is_valid_cell(start[0], start[1]):
        print(f"[A*] Start position {start} is not valid (obstacle or out of bounds)")
        return []
    
    # Validate goal position - if unreachable, will find closest cell instead
    if not is_valid_cell(goal[0], goal[1]):
        print(f"[A*] Goal position {goal} is not valid, will find closest reachable cell")
    
    # 8-directional movement
    movements = [
        (0, 1), (1, 0), (0, -1), (-1, 0),  # Straight
        (1, 1), (1, -1), (-1, -1), (-1, 1)  # Diagonal
    ]
    
    # Open set: heap of (f_score, counter, position)
    open_set = []
    closed_set = set()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    # Tie-breaker for heap stability
    counter = 0
    heapq.heappush(open_set, (f_score[start], counter, start))
    
    # Track closest cell to goal in case goal is unreachable
    closest_to_goal = start
    min_h_to_goal = heuristic(start, goal)
    
    while open_set:
        _, _, current = heapq.heappop(open_set)
        
        # Goal reached
        if current == goal:
            return reconstruct_path(came_from, current)
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        # Update closest cell to goal for fallback
        h_to_goal = heuristic(current, goal)
        if h_to_goal < min_h_to_goal:
            min_h_to_goal = h_to_goal
            closest_to_goal = current
        
        # Explore neighbors
        for dx, dy in movements:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if neighbor in closed_set:
                continue
            
            if not is_valid_cell(neighbor[0], neighbor[1]):
                continue
            
            # Calculate tentative g_score with clearance-based cost
            movement_cost = calculate_cost(current, neighbor, grid_map)
            tentative_g = g_score[current] + movement_cost
            
            # If this is a better path to the neighbor, update it
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
    
    # Goal unreachable, return path to closest reachable cell
    if closest_to_goal != start:
        print(f"[A*] Goal unreachable. Path leads to closest cell at {closest_to_goal} (distance: {min_h_to_goal:.2f})")
        return reconstruct_path(came_from, closest_to_goal)
    
    # No path found at all
    print(f"[A*] No path found from {start} toward {goal}")
    return []

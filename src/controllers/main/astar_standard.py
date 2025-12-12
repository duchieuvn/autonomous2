"""
Standard A* pathfinding implementation.
Treats obstacles (value > 0.5) as non-traversable, everything else as free space.
Uses 8-directional movement (including diagonals).
"""

import heapq
import numpy as np
from math import sqrt


def runAStarSearch(grid_map, start_coords, goal_coords):
    """
    Standard A* pathfinding algorithm.
    
    Args:
        grid_map: 2D numpy array where 0 = free, 1 = obstacle
        start_coords: (x, y) starting position
        goal_coords: (x, y) goal position
    
    Returns:
        List of (x, y) tuples representing the path, or empty list if no path found
    """
    rows, cols = grid_map.shape
    
    def is_walkable(x, y):
        """Check if a cell is walkable (not an obstacle)"""
        if not (0 <= x < cols and 0 <= y < rows):
            return False
        # Treat any value > 0.5 as obstacle, everything else as free
        return grid_map[y, x] < 0.5
    
    def heuristic(pos, goal):
        """Euclidean distance heuristic"""
        dx = pos[0] - goal[0]
        dy = pos[1] - goal[1]
        return sqrt(dx*dx + dy*dy)
    
    start = tuple(start_coords)
    goal = tuple(goal_coords)
    
    # Check if start and goal are valid
    if not is_walkable(start[0], start[1]):
        return []
    if not is_walkable(goal[0], goal[1]):
        return []
    
    # 8-directional movement (including diagonals)
    movements = [
        (0, 1),   # up
        (1, 0),   # right
        (0, -1),  # down
        (-1, 0),  # left
        (1, 1),   # up-right
        (1, -1),  # down-right
        (-1, -1), # down-left
        (-1, 1),  # up-left
    ]
    
    open_set = []
    closed_set = set()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    # Use counter for tie-breaking in heap
    counter = 0
    heapq.heappush(open_set, (f_score[start], counter, start))
    
    while open_set:
        current = heapq.heappop(open_set)[2]
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        closed_set.add(current)
        
        for dx, dy in movements:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if neighbor in closed_set:
                continue
            
            if not is_walkable(neighbor[0], neighbor[1]):
                continue
            
            # Calculate tentative g_score
            if abs(dx) + abs(dy) == 1:
                # Straight movement
                tentative_g = g_score[current] + 1.0
            else:
                # Diagonal movement (slightly longer)
                tentative_g = g_score[current] + sqrt(2)
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
    
    # No path found
    return []

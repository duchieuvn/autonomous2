import heapq
import numpy as np

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
            self.entry_count = 0  # used for tie-breaking

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
            return heapq.heappop(self.open_heap)[2]  # return the Point only

    def tracePath(goal):
        path = []
        current = goal
        while current.parent is not None:
            path.append((current.x, current.y))
            current = current.parent
        path.append((current.x, current.y))
        return path[::-1]

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
                    continue  # already pushed in updateCost
        current = frontier.pop()

    if current is None:
        print("Visited cells:", frontier.visited_count)
        print("No path found")
        return []

    path = tracePath(current)
    return path

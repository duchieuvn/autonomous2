"""Clean visualizer for grid maps with support for paths and points.

Maps grid values to custom colors and overlays paths/points for debugging.
"""
import pygame
import numpy as np
import cv2
from CONSTANTS import *


class MapVisualizer:
    """Visualize occupancy grid maps with custom color mapping and overlays."""

    def __init__(self, window_size=(800, 800)):
        """Initialize visualizer.

        Args:
            map_size: Grid dimension (map_size x map_size)
            window_size: Display window size (width, height)
            color_map: Dict mapping grid values to RGB colors.
                      Example: {0: (0,0,0), 100: (255,255,255), 255: (80,80,80)}
        """
        self.map_size = MAP_SIZE
        self.window_size = window_size
        self.color_map = self._default_color_map()
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Grid Map Visualizer")
        self.clock = pygame.time.Clock()

    def _default_color_map(self):
        """Default color mapping for common grid values."""
        return {
            0: (255, 255, 255),   # freespace
            1: (0, 0, 0),         # obstacle
            255: (80, 80, 80),    # unknown
            100: (0, 255, 0),     # start: green
            150: (0, 0, 255),     # end: blue
            180: (0, 255, 255),   # frontier generic: cyan
            # Frontier size gradient (values defined in CONSTANTS)
            50: (0, 0, 255),      # small: blue
            101: (0, 255, 255),   # medium: cyan (distinct from START_POINT_VALUE)
            200: (255, 255, 0),   # large: yellow
            220: (255, 0, 0),     # largest: red
        }

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

    def draw_path(self, path, color=(255, 0, 0), thickness=2):
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
            pygame.draw.line(self.screen, color, scaled_path[i], scaled_path[i + 1], thickness)

    def draw_point(self, map_x, map_y, color=(0, 255, 0), radius=5):
        """Draw a single point on screen.

        Args:
            map_x, map_y: Map coordinates
            color: RGB tuple for point color
            radius: Circle radius in pixels
        """
        screen_x, screen_y = self._map_to_screen(map_x, map_y)
        pygame.draw.circle(self.screen, color, (screen_x, screen_y), radius)

    def _draw_points(self, points, color=(0, 255, 0), radius=5):
        """Draw multiple points on screen.

        Args:
            points: List of (x, y) map coordinates
            color: RGB tuple for points color
            radius: Circle radius in pixels
        """
        for x, y in points:
            self.draw_point(x, y, color, radius)

    def display(self, grid_map, path=None):
        """Display the grid map with optional overlays.

        Args:
            grid_map: 2D numpy array with grid values
            path: Optional list of (x, y) map coordinates for path
            points: Optional list of (x, y) map coordinates for points
            point_color: RGB tuple for points color
            point_radius: Radius of point circles
        """
        path_color=(255, 0, 0)
        # Convert grid to display image
        display_img = self._grid_to_display(grid_map)

        # Resize and create surface
        resized = cv2.resize(display_img, self.window_size, interpolation=cv2.INTER_NEAREST)
        surface = pygame.surfarray.make_surface(np.transpose(resized, (1, 0, 2)))
        self.screen.blit(surface, (0, 0))

        # Draw overlays
        if path:
            self.draw_path(path, path_color, thickness=2)

        # pygame.display.flip() # This should be called outside, after all drawing is done.

    def clear(self):
        """Clear the screen."""
        self.screen.fill((0, 0, 0))
        pygame.display.flip()

    def close(self):
        """Close the visualizer window."""
        pygame.quit()

    def handle_events(self):
        """Handle pygame events. Returns True if window should close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False
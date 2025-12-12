import numpy as np
import cv2 

def get_angle_diff(a, b):
    diff = a - b
    while diff > np.pi:
        diff -= 2*np.pi
    while diff < -np.pi:
        diff += 2*np.pi
    return diff

def remove_noisy_pixels(grid_map, obstacle_value=1, connectivity=8):
    """
    Removes isolated obstacle pixels (single pixels with no neighbors).
    """
    binary_uint8 = np.where(grid_map == obstacle_value, 255, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_uint8, connectivity=connectivity
    )
    cleaned = np.zeros_like(binary_uint8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] > 1:
            cleaned[labels == label] = 255
    cleaned_map = np.where(cleaned == 255, obstacle_value, 0)
    return cleaned_map

def inflate_obstacles(grid_map, inflation_pixels=10):
    kernel_size = int(2 * inflation_pixels + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    grid_uint8 = np.where(grid_map == 1, 255, 0).astype(np.uint8)
    inflated = cv2.dilate(grid_uint8, kernel, iterations=1)
    inflated_map = (inflated > 0).astype(np.uint8)
    return inflated_map

def expand_free_pixel(map_array, map_point, inflation_pixels=1):
    # Purpose: 
    # The expand_free_pixel function expands a single map cell designated as free space 
    # (at the position map_point) into a circular region within the map_array, 
    # setting all included cells to free (typically value 0). 
    # This is used to guarantee that path endpoints (start/goal) are accessible
    # when obstacle inflation would otherwise make these points impassable for planners.
    # The argument inflation_pixels controls the radius of the free area created.
    h, w = map_array.shape
    x = map_point[0]
    y = map_point[1]
    for dy in range(-inflation_pixels, inflation_pixels + 1):
        for dx in range(-inflation_pixels, inflation_pixels + 1):
            if dx**2 + dy**2 <= inflation_pixels**2:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    map_array[ny, nx] = 0  # FREESPACE_VALUE

def bresenham_line(start, end):
    x1, y1 = start
    x2, y2 = end
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        err2 = err * 2
        if err2 > -dy:
            err -= dy
            x1 += sx
        if err2 < dx:
            err += dx
            y1 += sy
    return points

def boundary_laplacian(grid_map):
    img = (grid_map * 255).astype(np.uint8)
    lap = cv2.Laplacian(img, cv2.CV_8U, ksize=3)
    return (lap > 0).astype(np.uint8)

def boundary_density(boundary_mask, r=3):
    """
    return 2d density matrix
    """
    kernel = np.ones((2*r+1, 2*r+1), dtype=np.uint8)

    density = cv2.filter2D(boundary_mask.astype(np.uint8), -1, kernel)
    return density

def top_frontier_positions(density, top_k = 3):
    flat_indices = np.argpartition(density.ravel(), -top_k)[-top_k:]
    coords_2d = np.array(np.unravel_index(flat_indices, density.shape)).T # [(row, col), (y, x), ...]
    coords_2d = coords_2d[:, ::-1]
    return coords_2d

def select_nearest_high_density(density, map_position, top_k=3):
    flat_indices = np.argpartition(density.ravel(), -top_k)[-top_k:]
    coords = np.array(np.unravel_index(flat_indices, density.shape)).T
    
    px, py = map_position
    distances = [np.linalg.norm([x - px, y - py]) for y, x in coords]
    
    nearest_idx = np.argmin(distances)
    return tuple(coords[nearest_idx])

def segment_color(hsv_img, color):
    """Segments an image based on the specified color in HSV space."""
    if color == 'red':
        # Red can wrap around in HSV, so we need two ranges
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)

        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)

        return cv2.bitwise_or(mask1, mask2)

    elif color == 'blue':
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
        return blue_mask

    elif color == 'yellow':
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
        return yellow_mask

    elif color == 'green':
        lower_green = np.array([36, 100, 100])
        upper_green = np.array([86, 255, 255])
        green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
        return green_mask

    return None

def red_is_middle(hsv_img):
    "check if hsv_img has a red square in the middle"
    red_mask = segment_color(hsv_img, 'red')

    # If there are no red pixels at all, it's not in the middle.
    if cv2.countNonZero(red_mask) == 0:
        return False, red_mask

    height, width, _ = hsv_img.shape
    margin = 30

    # Check for red pixels on the border
    top_border = red_mask[0:margin, :]
    bottom_border = red_mask[height-margin:height, :]
    left_border = red_mask[:, 0:margin]
    right_border = red_mask[:, width-margin:width]

    # If any red is found on any border, it's not fully in the middle.
    if np.any(top_border) or np.any(bottom_border) or np.any(left_border) or np.any(right_border):
        return False, red_mask

    # If red is present but not on the margins, it's in the middle.
    return True, red_mask

def percentage_map_differences(map1, map2):
    # Count number of different pixels
    diff_count = np.sum(map1 != map2)
    return diff_count / (map1.shape[0] * map1.shape[1])

def save_map(map, save_path):
    bw = (map * 255).astype(np.uint8)
    cv2.imwrite(save_path, bw) 
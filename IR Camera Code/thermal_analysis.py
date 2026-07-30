import os
import cv2
import numpy as np
import json

def save_background(background_frame, filename="background.npy"):
    """
    Saves the captured thermal background array to a binary .npy file for later use.
    """
    if background_frame is not None:
        np.save(filename, background_frame)
        print(f"Background saved successfully to {filename}")

def load_background(filename="background.npy"):
    """
    Loads the thermal background array from a .npy file.
    Returns None if the file does not exist.
    """
    if os.path.exists(filename):
        print(f"Loading background from {filename}...")
        return np.load(filename)
    print(f"Background file {filename} not found.")
    return None

def get_hot_cold_spots(temp_array, size=1):
    """
    Analyzes a 2D temperature array to find the hottest and coldest areas of pixels of size (size x size).
    Returns the temperatures and their (X, Y) coordinates.
    """
    for i in range(size):
        for j in range(size):
            if i == 0 and j == 0:
                continue
            temp_array = np.maximum(temp_array, np.roll(temp_array, shift=(i, j), axis=(0, 1)))
            temp_array = np.minimum(temp_array, np.roll(temp_array, shift=(-i, -j), axis=(0, 1)))
    max_temp = np.max(temp_array)
    min_temp = np.min(temp_array)
    
    max_y, max_x = np.unravel_index(np.argmax(temp_array), temp_array.shape)
    min_y, min_x = np.unravel_index(np.argmin(temp_array), temp_array.shape)
    
    return (max_temp, max_x, max_y), (min_temp, min_x, min_y)

def get_hot_cold_spots_with_threshold(temp_array, size=1, high_threshold=0, low_threshold=0):
    """
    Analyzes a 2D temperature array to find the hottest and coldest areas of pixels of size (size x size).
    Returns the temperatures and their (X, Y) coordinates, but only if they exceed a certain threshold.
    """
    hot_data, cold_data = get_hot_cold_spots(temp_array, size)
    
    if hot_data[0] < high_threshold:
        hot_data = (None, None, None)
    
    if cold_data[0] > low_threshold:
        cold_data = (None, None, None)
    
    return hot_data, cold_data

def subtract_background(current_frame, background_frame):
    """
    Subtracts a static thermal baseline (e.g., warm stepper motors) from the current frame.
    This eliminates static heat sources and isolates the calibration target.
    """
    if background_frame is None:
        return current_frame
    
    clean_frame = current_frame - background_frame
    return np.clip(clean_frame, a_min=0, a_max=None)

def get_hot_spot_centroid(temp_array, threshold=23.0):
    """
    Calculates the sub-pixel centroid of the hottest region above a given threshold.
    If threshold is less than 1.0 (e.g., 0.75), it is treated as a relative fraction of the maximum value.
    Otherwise, it is treated as an absolute temperature threshold.
    Returns the max temperature and the (X, Y) sub-pixel coordinates.
    """
    temp_array_float = temp_array.astype(np.float32)
    max_val = np.max(temp_array_float)
    
    if threshold < 1.0:
        actual_threshold = max_val * threshold
        if max_val < 3.0: 
            return None, None, None
    else:
        actual_threshold = threshold
    
    _, mask = cv2.threshold(temp_array_float, actual_threshold, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    
    M = cv2.moments(mask)
    
    if M["m00"] != 0:
        c_x = M["m10"] / M["m00"]
        c_y = M["m01"] / M["m00"]
        
        max_temp = np.max(temp_array[mask == 255])
        return max_temp, c_x, c_y
        
    return None, None, None

def calibrate_camera_perspective(pixel_points, mm_points, filename="transform_matrix.json"):
    """
    Calculates a 3x3 transformation matrix to convert pixels to mm, 
    accounting for camera tilt and perspective distortion.
    """
    pts_pixel = np.array(pixel_points, dtype=np.float32)
    pts_mm = np.array(mm_points, dtype=np.float32)
    
    pixel_points_avg = np.mean(pts_pixel, axis=0)
    print(f"Average pixel points: {pixel_points_avg}")
    
    matrix = cv2.getPerspectiveTransform(pixel_points_avg, pts_mm)
    
    if os.path.exists(filename):
        os.remove(filename)
        
    with open(filename, "w") as f:
        json.dump(matrix.tolist(), f)
        
    return matrix

def get_mm_from_pixels(pixel_x, pixel_y, matrix):
    """
    Converts a single (x, y) pixel coordinate to mm using the provided transformation matrix.
    
    Parameters:
    - pixel_x: The x-coordinate in pixels.
    - pixel_y: The y-coordinate in pixels.
    - matrix: A 3x3 numpy array representing the transformation matrix.
    
    Returns:
    - mm_x: The x-coordinate in millimeters.
    - mm_y: The y-coordinate in millimeters.
    """
    pt_pixel = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    pt_mm = cv2.perspectiveTransform(pt_pixel, matrix)
    mm_x, mm_y = pt_mm[0][0]
    
    return mm_x, mm_y

def load_transform_matrix(filename="transform_matrix.json"):
    """
    Loads the transformation matrix from a JSON file.
    """
    with open(filename, "r") as f:
        matrix = np.array(json.load(f), dtype=np.float32)
    return matrix
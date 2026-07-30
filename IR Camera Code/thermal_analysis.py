import os
import cv2
import numpy as np
import json

def get_hot_cold_spots(temp_array, size=1):
    """
    Analyzes a 2D temperature array to find the hottest and coldest areas of pixels of size (size x size).
    Returns the temperatures and their (X, Y) coordinates.
    """
    # Find the maximum and minimum temperature values
    for i in range(size): # Average over a square of pixels of size (size x size)
        for j in range(size):
            # Create a mask to ignore the edges of the array when looking for max/min
            if i == 0 and j == 0:
                continue
            temp_array = np.maximum(temp_array, np.roll(temp_array, shift=(i, j), axis=(0, 1)))
            temp_array = np.minimum(temp_array, np.roll(temp_array, shift=(-i, -j), axis=(0, 1)))
    max_temp = np.max(temp_array)
    min_temp = np.min(temp_array)
    
    # Find the linear index of the max and min values, then convert them 
    # back into 2D (Y, X) coordinates based on the array's shape
    max_y, max_x = np.unravel_index(np.argmax(temp_array), temp_array.shape)
    min_y, min_x = np.unravel_index(np.argmin(temp_array), temp_array.shape)
    
    # Return as two tuples: (temperature, x_coord, y_coord)
    return (max_temp, max_x, max_y), (min_temp, min_x, min_y)

def get_hot_cold_spots_with_threshold(temp_array, size=1, high_threshold=0, low_threshold=0):
    """
    Analyzes a 2D temperature array to find the hottest and coldest areas of pixels of size (size x size).
    Returns the temperatures and their (X, Y) coordinates, but only if they exceed a certain threshold.
    """
    hot_data, cold_data = get_hot_cold_spots(temp_array, size)
    
    # Check if the hottest temperature exceeds the threshold
    if hot_data[0] < high_threshold:
        hot_data = (None, None, None)  # Set to None if below threshold
    
    # Check if the coldest temperature is below the negative threshold
    if cold_data[0] > low_threshold:
        cold_data = (None, None, None)  # Set to None if above negative threshold
    
    return hot_data, cold_data

def subtract_background(current_frame, background_frame):
    """
    Subtracts a static thermal baseline (e.g., warm stepper motors) from the current frame.
    This eliminates static heat sources and isolates the calibration target.
    """
    if background_frame is None:
        return current_frame
    
    # Subtract the static baseline; clip at 0 to prevent negative values 
    # from artificially lowering the temperature of the target area
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
    
    # If using relative thresholding
    if threshold < 1.0:
        actual_threshold = max_val * threshold
        # Require a minimum signal to avoid thresholding background noise
        if max_val < 3.0: 
            return None, None, None
    else:
        actual_threshold = threshold
    
    # Create a binary mask of pixels above the temperature threshold
    _, mask = cv2.threshold(temp_array_float, actual_threshold, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    
    # Calculate image moments to find the center of mass of the heat signature
    M = cv2.moments(mask)
    
    # Ensure the area (m00) is not zero to prevent division by zero errors
    if M["m00"] != 0:
        # Calculate precise sub-pixel coordinates
        c_x = M["m10"] / M["m00"]
        c_y = M["m01"] / M["m00"]
        
        # Get the actual max temperature within the isolated region
        max_temp = np.max(temp_array[mask == 255])
        return max_temp, c_x, c_y
        
    return None, None, None

def calibrate_camera_perspective(pixel_points, mm_points, filename="transform_matrix.json"):
    """
    Calculates a 3x3 transformation matrix to convert pixels to mm, 
    accounting for camera tilt and perspective distortion.
    
    Inputs:
        pixel_points: List of 4 [x, y] pixel coordinates (e.g., [[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        mm_points: List of the corresponding 4 [x, y] coordinates in mm on the gantry
    Outputs:
        matrix: The 3x3 transformation matrix (also saved to a JSON file)
    """

    # OpenCV requires float32 numpy arrays for this calculation
    pts_pixel = np.array(pixel_points, dtype=np.float32)
    pts_mm = np.array(mm_points, dtype=np.float32)
    #average the 8 pairs of pixel points to get 4 points for the perspective transform
    pixel_points_avg = np.mean(pts_pixel, axis=0)
    print(f"Average pixel points: {pixel_points_avg}")
    # Calculate the 3x3 perspective transform matrix
    # This matrix mathematically maps the pixel quadrilateral to the physical mm rectangle
    matrix = cv2.getPerspectiveTransform(pixel_points_avg, pts_mm)
    # Check if the filen
    #  already exists, if so, overwrite it
    
    if os.path.exists(filename):
        os.remove(filename)
    # Store the 3x3 matrix in a JSON file for later use
    with open(filename, "w") as f:
        json.dump(matrix.tolist(), f)
        
    return matrix

def get_mm_from_pixels(pixel_x, pixel_y, matrix_filename="transform_matrix.json"):
    """
    Converts a single (x, y) pixel coordinate to mm using the saved transformation matrix.
    """
    # Load the matrix
    with open(matrix_filename, "r") as f:
        matrix = np.array(json.load(f), dtype=np.float32)
        
    # OpenCV expects an array of shape (N, 1, 2) for the transform
    pt_pixel = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    
    # Apply the perspective transformation
    pt_mm = cv2.perspectiveTransform(pt_pixel, matrix)
    
    # Extract the x and y millimeter coordinates
    mm_x, mm_y = pt_mm[0][0]
    
    return mm_x, mm_y

def load_transform_matrix(filename="transform_matrix.json"):
    """
    Loads the transformation matrix from a JSON file.
    """
    with open(filename, "r") as f:
        matrix = np.array(json.load(f), dtype=np.float32)
    return matrix
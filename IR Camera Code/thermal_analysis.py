import numpy as np

def get_hot_cold_spots(temp_array):
    """
    Analyzes a 2D temperature array to find the hottest and coldest pixels.
    Returns the temperatures and their (X, Y) coordinates.
    """
    # Find the maximum and minimum temperature values
    max_temp = np.max(temp_array)
    min_temp = np.min(temp_array)
    
    # Find the linear index of the max and min values, then convert them 
    # back into 2D (Y, X) coordinates based on the array's shape
    max_y, max_x = np.unravel_index(np.argmax(temp_array), temp_array.shape)
    min_y, min_x = np.unravel_index(np.argmin(temp_array), temp_array.shape)
    
    # Return as two tuples: (temperature, x_coord, y_coord)
    return (max_temp, max_x, max_y), (min_temp, min_x, min_y)
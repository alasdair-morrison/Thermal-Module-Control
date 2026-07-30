import os
import sys
print(sys.executable)
import PySpin
import matplotlib
matplotlib.use('Qt5Agg')  # Add this line to force an interactive window
import matplotlib.pyplot as plt
import keyboard
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import ir_camera_code.thermal_analysis as ta
from zaber_motion import Units
from zaber_motion.ascii import Connection
import threading
import time
import math

# Global state for cross-thread communication
GLOBAL_HOT_SPOT = None
GLOBAL_COLD_SPOT = None
SPOTS_AVAILABLE = False
CONTINUE_RECORDING = True

def move_heater_to_target(x, y, target_x, target_y):
    """
    Moves the heater to the target position in absolute coordinates
    """
    x.move_absolute(target_x, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
    y.move_absolute(target_y, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
    x.wait_until_idle()
    y.wait_until_idle()

def gantry_worker(x, y):
    """
    Runs on a background thread. Executes a central spiral, 
    then actively evades hot spots and hunts cold spots.
    """
    global CONTINUE_RECORDING, GLOBAL_HOT_SPOT, GLOBAL_COLD_SPOT, SPOTS_AVAILABLE
    
    center_x, center_y = 125.0, 125.0
    max_radius = 120.0  # Stop 5mm short of the physical 250mm edge
    pitch = 15.0        # 15mm outward expansion per 360-degree revolution
    theta = 0.0
    current_x, current_y = center_x, center_y
    
    print("Gantry: Starting Spiral Phase...")
    
    while CONTINUE_RECORDING:
        r = (pitch / (2 * math.pi)) * theta
        if r > max_radius:
            print("Gantry: Edge reached. Switching to Reactive Phase...")
            break  # Exit spiral loop
            
        current_x = center_x + r * math.cos(theta)
        current_y = center_y + r * math.sin(theta)
        
        move_heater_to_target(x, y, current_x, current_y)
        theta += 0.5  # Increment angle by roughly 28 degrees per step
        
    step_size = 10.0  # Move 10mm per reactive adjustment
    
    while CONTINUE_RECORDING:
        if not SPOTS_AVAILABLE:
            time.sleep(0.1)
            continue
            
        # Safely grab the latest coordinates from the camera thread
        hx, hy = GLOBAL_HOT_SPOT
        cx, cy = GLOBAL_COLD_SPOT
        
        # Vector pointing AWAY from the hot spot
        v_hot_x = current_x - hx
        v_hot_y = current_y - hy
        mag_hot = math.hypot(v_hot_x, v_hot_y)
        if mag_hot > 0:
            v_hot_x /= mag_hot
            v_hot_y /= mag_hot
            
        # Vector pointing TOWARD the cold spot
        v_cold_x = cx - current_x
        v_cold_y = cy - current_y
        mag_cold = math.hypot(v_cold_x, v_cold_y)
        if mag_cold > 0:
            v_cold_x /= mag_cold
            v_cold_y /= mag_cold
            
        # Combine vectors to find the ideal path
        dir_x = v_hot_x + v_cold_x
        dir_y = v_hot_y + v_cold_y
        mag_dir = math.hypot(dir_x, dir_y)
        
        # Normalize the combined vector and scale by step_size
        if mag_dir > 0:
            dir_x = (dir_x / mag_dir) * step_size
            dir_y = (dir_y / mag_dir) * step_size
            
        next_x = current_x + dir_x
        next_y = current_y + dir_y
        
        # Hard boundaries (clamp to the 0-250mm physical workspace)
        next_x = max(0.0, min(250.0, next_x))
        next_y = max(0.0, min(250.0, next_y))
        
        # Move and wait
        move_heater_to_target(x, y, next_x, next_y)
        current_x, current_y = next_x, next_y
        
        # Dwell briefly to let the material absorb heat and update the thermal signature
        time.sleep(0.5)

class IRFormatType:
    LINEAR_10MK = 1
    LINEAR_100MK = 2
    RADIOMETRIC = 3

CHOSEN_IR_TYPE = IRFormatType.RADIOMETRIC


def handle_close(evt):
    """
    This function will close the GUI when close event happens.

    :param evt: Event that occurs when the figure closes.
    :type evt: Event
    """
    global CONTINUE_RECORDING
    CONTINUE_RECORDING = False


def acquire_and_display_images(cam, nodemap, nodemap_tldevice):
    """
    This function continuously acquires images from a device and display them in a GUI.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    global CONTINUE_RECORDING

    sNodemap = cam.GetTLStreamNodeMap()

    # Change bufferhandling mode to NewestOnly
    node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))

    node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
    node_pixel_format_mono16 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName('Mono16'))
    pixel_format_mono16 = node_pixel_format_mono16.GetValue()
    node_pixel_format.SetIntValue(pixel_format_mono16)

    if CHOSEN_IR_TYPE == IRFormatType.LINEAR_10MK:
        # This section is to be activated only to set the streaming mode to TemperatureLinear10mK
        node_IRFormat = PySpin.CEnumerationPtr(nodemap.GetNode('IRFormat'))
        node_temp_linear_high = PySpin.CEnumEntryPtr(node_IRFormat.GetEntryByName('TemperatureLinear10mK'))
        node_temp_high = node_temp_linear_high.GetValue()
        node_IRFormat.SetIntValue(node_temp_high)
    elif CHOSEN_IR_TYPE == IRFormatType.LINEAR_100MK:
        # This section is to be activated only to set the streaming mode to TemperatureLinear100mK
        node_IRFormat = PySpin.CEnumerationPtr(nodemap.GetNode('IRFormat'))
        node_temp_linear_low = PySpin.CEnumEntryPtr(node_IRFormat.GetEntryByName('TemperatureLinear100mK'))
        node_temp_low = node_temp_linear_low.GetValue()
        node_IRFormat.SetIntValue(node_temp_low)
    elif CHOSEN_IR_TYPE == IRFormatType.RADIOMETRIC:
        # This section is to be activated only to set the streaming mode to Radiometric
        node_IRFormat = PySpin.CEnumerationPtr(nodemap.GetNode('IRFormat'))
        node_temp_radiometric = PySpin.CEnumEntryPtr(node_IRFormat.GetEntryByName('Radiometric'))
        node_radiometric = node_temp_radiometric.GetValue()
        node_IRFormat.SetIntValue(node_radiometric)

    if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
        print('Unable to set stream buffer handling mode.. Aborting...')
        return False

    # Retrieve entry node from enumeration node
    node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
    if not PySpin.IsAvailable(node_newestonly) or not PySpin.IsReadable(node_newestonly):
        print('Unable to set stream buffer handling mode.. Aborting...')
        return False

    # Retrieve integer value from entry node
    node_newestonly_mode = node_newestonly.GetValue()

    # Set integer value from entry node as new value of enumeration node
    node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

    print('*** IMAGE ACQUISITION ***\n')
    try:
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition mode set to continuous...')

        #  Begin acquiring images
        #
        #  *** NOTES ***
        #  What happens when the camera begins acquiring images depends on the
        #  acquisition mode. Single frame captures only a single image, multi
        #  frame catures a set number of images, and continuous captures a
        #  continuous stream of images.
        #
        #  *** LATER ***
        #  Image acquisition must be ended when no more images are needed.
        cam.BeginAcquisition()

        print('Acquiring images...')

        #  Retrieve device serial number for filename
        #
        #  *** NOTES ***
        #  The device serial number is retrieved in order to keep cameras from
        #  overwriting one another. Grabbing image IDs could also accomplish
        #  this.
        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        # Retrieve Calibration details
        CalibrationQueryR_node = PySpin.CFloatPtr(nodemap.GetNode('R'))
        R = CalibrationQueryR_node.GetValue()
        print('R =', R)

        CalibrationQueryB_node = PySpin.CFloatPtr(nodemap.GetNode('B'))
        B = CalibrationQueryB_node.GetValue()
        print('B =', B)

        CalibrationQueryF_node = PySpin.CFloatPtr(nodemap.GetNode('F'))
        F = CalibrationQueryF_node.GetValue()
        print('F =', F)

        CalibrationQueryX_node = PySpin.CFloatPtr(nodemap.GetNode('X'))
        X = CalibrationQueryX_node.GetValue()
        print('X =', X)

        CalibrationQueryA1_node = PySpin.CFloatPtr(nodemap.GetNode('alpha1'))
        A1 = CalibrationQueryA1_node.GetValue()
        print('alpha1 =', A1)

        CalibrationQueryA2_node = PySpin.CFloatPtr(nodemap.GetNode('alpha2'))
        A2 = CalibrationQueryA2_node.GetValue()
        print('alpha2 =', A2)

        CalibrationQueryB1_node = PySpin.CFloatPtr(nodemap.GetNode('beta1'))
        B1 = CalibrationQueryB1_node.GetValue()
        print('beta1 =', B1)

        CalibrationQueryB2_node = PySpin.CFloatPtr(nodemap.GetNode('beta2'))
        B2 = CalibrationQueryB2_node.GetValue()
        print('beta2 =', B2)

        CalibrationQueryJ1_node = PySpin.CFloatPtr(nodemap.GetNode('J1'))    # Gain
        J1 = CalibrationQueryJ1_node.GetValue()
        print('Gain =', J1)

        CalibrationQueryJ0_node = PySpin.CIntegerPtr(nodemap.GetNode('J0'))   # Offset
        J0 = CalibrationQueryJ0_node.GetValue()
        print('Offset =', J0)

        # Figure(1) is default so you can omit this line. Figure(0) will create a new window every time program hits this line
        fig = plt.figure(1)

        # Close the GUI when close event happens
        fig.canvas.mpl_connect('close_event', handle_close)

        if CHOSEN_IR_TYPE == IRFormatType.RADIOMETRIC:
            # Object Parameters. For this demo, they are imposed!
            # This section is important when the streaming is set to radiometric and not TempLinear
            # Image of temperature is calculated computer-side and not camera-side
            # Parameters can be set to the whole image, or for a particular ROI (not done here)
            Emiss = 0.97
            TRefl = 293.15
            TAtm = 293.15
            TAtmC = TAtm - 273.15
            Humidity = 0.55

            Dist = 2
            ExtOpticsTransmission = 1
            ExtOpticsTemp = TAtm

            H2O = Humidity * np.exp(1.5587 + 0.06939 * TAtmC - 0.00027816 * TAtmC * TAtmC + 0.00000068455 * TAtmC * TAtmC * TAtmC)
            print('H20 =', H2O)

            Tau = X * np.exp(-np.sqrt(Dist) * (A1 + B1 * np.sqrt(H2O))) + (1 - X) * np.exp(-np.sqrt(Dist) * (A2 + B2 * np.sqrt(H2O)))
            print('tau =', Tau)

            # Pseudo radiance of the reflected environment
            r1 = ((1 - Emiss) / Emiss) * (R / (np.exp(B / TRefl) - F))
            print('r1 =', r1)

            # Pseudo radiance of the atmosphere
            r2 = ((1 - Tau) / (Emiss * Tau)) * (R / (np.exp(B / TAtm) - F))
            print('r2 =', r2)

            # Pseudo radiance of the external optics
            r3 = ((1 - ExtOpticsTransmission) / (Emiss * Tau * ExtOpticsTransmission)) * (R / (np.exp(B / ExtOpticsTemp) - F))
            print('r3 =', r3)

            K2 = r1 + r2 + r3
            print('K2 =', K2)
        if os.path.exists("background.npy"):
            background_Temp = ta.load_background(filename="background.npy")  # Load the background frame from a .npy file if it exists
        else:
            print("No background frame found. Please run the calibration script first.")
            background_Temp = None
            return False
        if os.path.exists("transform_matrix.json"):
                                    transform_matrix = ta.load_transform_matrix("transform_matrix.json")
        else:
            print("No transform matrix found. Please run the calibration script first.")
            transform_matrix = None
            return False
        # Retrieve and display images
        print('Press Enter to stop streaming')
        while(CONTINUE_RECORDING):
            try:

                #  Retrieve next received image
                #
                #  *** NOTES ***
                #  Capturing an image houses images on the camera buffer. Trying
                #  to capture an image that does not exist will hang the camera.
                #
                #  *** LATER ***
                #  Once an image from the buffer is saved and/or no longer
                #  needed, the image must be released in order to keep the
                #  buffer from filling up.

                image_result = cam.GetNextImage()

                #  Ensure image completion
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                else:

                    # Getting the image data as a np array
                    image_data = image_result.GetNDArray()

                    # Adapt the title to the correct streaming mode: TempLinear10mK, or TempLinear100mK or pseudo Radiance or Temperature Radiometric
                    fig.suptitle('A700 Temperature Radiometric')

                    if CHOSEN_IR_TYPE == IRFormatType.RADIOMETRIC:
                        global GLOBAL_HOT_SPOT, GLOBAL_COLD_SPOT, SPOTS_AVAILABLE
                        
                        image_Radiance = (image_data - J0) / J1
                        image_Temp = (B / np.log(R / ((image_Radiance / Emiss / Tau) - K2) + F)) - 273.15
                        
                        
                        clean_temp_array = ta.subtract_background(image_Temp, background_Temp)
                        hot_max, h_px_x, h_px_y = ta.get_hot_spot_centroid(clean_temp_array, threshold=0.75)
                        
                        # Find the absolute minimum temperature in the frame directly
                        cold_min, c_px_x, c_px_y = ta.get_cold_spot_centroid(image_Temp, threshold=0.15)

                        if hot_max is not None or cold_min is not None:
                            SPOTS_AVAILABLE = True
                            
                            if hot_max is not None:
                                hot_mm_x, hot_mm_y = ta.transform_coordinates(h_px_x, h_px_y, transform_matrix)
                                GLOBAL_HOT_SPOT = (hot_mm_x, hot_mm_y)
                                plt.plot(h_px_x, h_px_y, marker='+', color='red', markersize=15)
                            else:
                                GLOBAL_HOT_SPOT = None

                            if cold_min is not None:
                                cold_mm_x, cold_mm_y = ta.transform_coordinates(c_px_x, c_px_y, transform_matrix)
                                GLOBAL_COLD_SPOT = (cold_mm_x, cold_mm_y)
                                plt.plot(c_px_x, c_px_y, marker='+', color='cyan', markersize=15)
                            else:
                                GLOBAL_COLD_SPOT = None
                        else:
                            SPOTS_AVAILABLE = False
                            GLOBAL_HOT_SPOT = None
                            GLOBAL_COLD_SPOT = None
                            print("No valid hot or cold spots detected in this frame.")

                    # Interval in plt.pause(interval) determines how fast the images are displayed in a GUI
                    # Interval is in seconds.
                    plt.pause(0.001)

                    # Clear current reference of a figure. This will improve display speed significantly
                    plt.clf()

                    # If user presses enter, close the program
                    if keyboard.is_pressed('ENTER'):
                        print('Program is closing...')

                        # Close figure
                        plt.close('all')
                        CONTINUE_RECORDING = False

                #  Release image
                #
                #  *** NOTES ***
                #  Images retrieved directly from the camera (i.e. non-converted
                #  images) need to be released in order to keep from filling the
                #  buffer.
                image_result.Release()

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        #  End acquisition
        #
        #  *** NOTES ***
        #  Ending acquisition appropriately helps ensure that devices clean up
        #  properly and do not need to be power-cycled to maintain integrity.
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return True


def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True

        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Acquire images
        result &= acquire_and_display_images(cam, nodemap, nodemap_tldevice)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def main():
    """
    Example entry point; please see Enumeration example for more in-depth
    comments on preparing and cleaning up the system.

    :return: True if successful, False otherwise.
    :rtype: bool
    """
    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    # Run example on each camera
    for i, cam in enumerate(cam_list):

        print('Running example for camera %d...' % i)
        # Connect to Gantry and start the gantry worker thread
        with Connection.open_serial_port("COM6") as connection:
            connection.enable_alerts()
            device_list = connection.detect_devices()
            device = device_list[0]
            
            x = device.get_axis(1)
            y = device.get_axis(2)
            
            # Home the axes if necessary
            if not x.is_homed():
                x.home(wait_until_idle=False)
            if not y.is_homed():
                y.home(wait_until_idle=False)
            x.wait_until_idle()
            y.wait_until_idle()
            
            # daemon=True ensures the thread forcefully dies if the main program crashes
            gantry_thread = threading.Thread(target=hm.gantry_worker, args=(x, y), daemon=True)
            gantry_thread.start()
            
            # This will block the main thread and run continuously until the GUI is closed
            result &= run_single_camera(cam)
        print('Camera %d example complete... \n' % i)

    # Release reference to camera
    # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
    # cleaned up when going out of scope.
    # The usage of del is preferred to assigning the variable to None.
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    return result

if __name__ == '__main__':
    main()
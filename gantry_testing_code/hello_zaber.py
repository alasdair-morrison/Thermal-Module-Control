# Default sample code from Zaber Motion Library for Python
from zaber_motion import Units
from zaber_motion.ascii import Connection

with Connection.open_serial_port("COM6") as connection:
    connection.enable_alerts()

    device_list = connection.detect_devices()
    print("Found {} devices".format(len(device_list)))
    device = device_list[0]
    print("Device has {} axes".format(device.axis_count))
    axis = device.get_axis(1)
    # Home the axis if it is not already homed (just means check if the axis is at its reference position)
    if not axis.is_homed():
      print("Axis 1 is not homed. Homing now...")
      axis.home(wait_until_idle=False)
    axis2 = device.get_axis(2)
    if not axis2.is_homed():
      print("Axis 2 is not homed. Homing now...")
      axis2.home(wait_until_idle=False)
    axis.wait_until_idle()
    axis2.wait_until_idle()
    # Start from the base of each axis for demonstration consistency
    axis.move_absolute(0, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
    axis2.move_absolute(0, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
    axis.wait_until_idle()
    axis2.wait_until_idle()
      # Move to 10mm
    axis.move_absolute(10, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
    axis2.move_absolute(10, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
    axis.wait_until_idle()
    axis2.wait_until_idle()

    # Move by an additional 5mm
    axis.move_relative(15, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
    axis2.move_relative(15, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
    axis.wait_until_idle()
    axis2.wait_until_idle()
from zaber_motion import Units
from zaber_motion.ascii import Connection

with Connection.open_serial_port("COM6") as connection:
    connection.enable_alerts()

    device_list = connection.detect_devices()
    print("Found {} devices".format(len(device_list)))
    device = device_list[0]
    print("Device has {} axes".format(device.axis_count))
    axis = device.get_axis(1)
    axis2 = device.get_axis(2)
    axis.home(wait_until_idle=False)
    axis2.home(wait_until_idle=False)
    axis.wait_until_idle()
    axis2.wait_until_idle()
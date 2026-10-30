import time
from rosbridge_api import RosbridgeRobot

with RosbridgeRobot("192.168.1.213") as r:
    for _ in range(40):
        r.send_command(0.0, 0.5)
        time.sleep(0.05)
    r.send_command(0.0, 0.0)
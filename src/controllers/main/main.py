import pygame
from my_robot import MyRobot
import numpy as np

def main():
    robot = MyRobot()
    main_path = robot.explore()
    if not main_path:
        print("[warning] main_path is None/empty")
        return

    print("Found path")
    robot.step(100)
    robot.follow_final_path(main_path, debug_vis=True, replan_interval=60)



main()

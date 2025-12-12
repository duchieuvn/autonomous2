import pygame
from my_robot import MyRobot

import numpy as np

def main():
    robot = MyRobot()
    # Stagge 1
    main_path = robot.explore()
    print('Found path')

    # Stage 2
    cur_position = robot.get_map_position()
    first_path = robot.find_path(cur_position, robot.start_point)
    robot.step(100)
    robot.path_following_pipeline(first_path)
    robot.step(100)
    robot.path_following_pipeline(main_path)
    
main()  


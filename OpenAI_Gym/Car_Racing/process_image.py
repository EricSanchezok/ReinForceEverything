import cv2
import torch
import numpy as np

def get_vector(env, car_location):
    # 读取图像
    ori_img = cv2.cvtColor(np.array(env.render()), cv2.COLOR_RGB2BGR)

    # 检测车道边缘
    hsv = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 通过腐蚀和膨胀去除噪声
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 反转图像
    mask = cv2.bitwise_not(mask)

    # 把y>350的部分全部置为0
    mask[350:, :] = 0

    # cv2.imshow('mask', mask)

    # 设置一个以小车为中心的矩形视野范围
    view_rectange = [car_location[0]-150, car_location[1]-50, car_location[0]+150, car_location[1]]

    # 获取视野范围内的所有像素点
    view_pixels = mask[view_rectange[1]:view_rectange[3], view_rectange[0]:view_rectange[2]]

    # 计算视野范围内的白色像素点的重心
    M = cv2.moments(view_pixels)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

    # 计算小车中心到视野范围内白色像素点重心的向量
    vector = [cx+view_rectange[0]-car_location[0], cy+view_rectange[1]-car_location[1]]

    return ori_img, vector, cx, cy, view_rectange
    
def if_on_road(obs):
    ori_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)

    car_height = 12
    car_width = 6

    x_car = 48
    y_car = 72

    lower = np.array([40, 80, 130])
    upper = np.array([100, 160, 255])


    mask = cv2.inRange(ori_img, lower, upper)

    # 判断mask中的以x_car, y_car为中心的矩形中是否有白色像素
    for i in range(x_car - car_width // 2, x_car + car_width // 2):
        for j in range(y_car - car_height // 2, y_car + car_height // 2):
            if mask[j, i] == 255:
                return False

    return True

def cal_on_road_reward(obs):
    ori_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)

    cal_height = 32
    cal_width = 16

    x_car = 48
    y_car = 72

    grass_lower = np.array([40, 80, 130])
    grass_upper = np.array([100, 160, 255])
    grass_mask = cv2.inRange(ori_img, grass_lower, grass_upper)

    gray_lower = np.array([0, 0, 100]) 
    gray_upper = np.array([0, 0, 120])
    gray_mask = cv2.inRange(ori_img, gray_lower, gray_upper)

    grass_around_car = grass_mask[y_car - cal_height // 2 : y_car + cal_height // 2, x_car - cal_width // 2 : x_car + cal_width // 2]
    gray_around_car = gray_mask[y_car - cal_height // 2 : y_car + cal_height // 2, x_car - cal_width // 2 : x_car + cal_width // 2]

    grass_count = grass_around_car.sum()
    gray_count = gray_around_car.sum()

    # 计算比率,并限制范围在0-1之间
    coff = gray_count / (grass_count + gray_count)

    reward = 0.05 * coff


    return reward


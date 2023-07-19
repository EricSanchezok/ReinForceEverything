import gym
import cv2
import numpy as np
import process_image

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp  # 比例增益
        self.ki = ki  # 积分增益
        self.kd = kd  # 微分增益
        self.last_error = 0  # 上一次的误差
        self.integral = 0  # 积分项

    def compute_control(self, dis):
        error = dis  # 计算距离差异
        self.integral += error  # 累积误差，用于积分项

        # 当前误差与上一次误差的差值，用于微分项
        derivative = error - self.last_error

        # 计算输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # 将输出限制在 -1 到 1 之间
        output = max(-1, min(1, output))

        # 更新上一次的误差
        self.last_error = error
        return output



if __name__ == '__main__':

    cv2.namedWindow('image')

    car_location = [300, 300]
    car_rectange = [car_location[0]-15, car_location[1]-25, car_location[0]+15, car_location[1]+25]

    env = gym.make('CarRacing-v2', render_mode='rgb_array')
     
    # 初始化 PID 控制器
    angle_pid_controller = PIDController(kp=0.02, ki=0.0, kd=0.2)

    break_pid_controller = PIDController(kp=0.005, ki=0.0, kd=0.0)

    try_times = 10

    for time in range(try_times):

        continue_num = 0
        state, _ = env.reset()

        for step in range(10000):
            
            ori_img, vector, cx, cy, view_rectange = process_image.get_vector(env, car_location)

            # 计算pid控制器的输出
            angle = angle_pid_controller.compute_control(vector[0])

            # 计算vector的长度
            length = np.sqrt(vector[0]**2 + vector[1]**2)

            gas = 0.1
            length_error = length - 30

            breaking = break_pid_controller.compute_control(length_error)

            if abs(angle) > 0.1 and length_error > 0:
                gas = 0.0

            print("angle: ", round(angle, 3), "gas: ", round(gas, 3), "break: ", round(breaking, 3))

            action = [0.0, 0.0, 0.0]

            if step > 50:
                # 画出小车的位置
                cv2.rectangle(ori_img, (car_rectange[0], car_rectange[1]), (car_rectange[2], car_rectange[3]), (255, 0, 0), 2)

                # 画出矩形视野范围
                cv2.rectangle(ori_img, (view_rectange[0], view_rectange[1]), (view_rectange[2], view_rectange[3]), (0, 0, 255), 2)

                # 画出视野范围内的白色像素点的重心
                if cx != 0 and cy != 0:
                    cv2.circle(ori_img, (cx+view_rectange[0], cy+view_rectange[1]), 5, (255, 0, 255), -1)

                    # 画出小车中心到视野范围内白色像素点重心的向量
                    cv2.arrowedLine(ori_img, (car_location[0], car_location[1]), (cx+view_rectange[0], cy+view_rectange[1]), (0, 255, 0), 2)

                action = [angle, gas, breaking]

            cv2.imshow('image', ori_img)
            key = cv2.waitKey(1)

            state, reward, done, _, _ = env.step(action)

            if reward < 0:
                continue_num += 1
            else:
                continue_num = 0


            if done or key == ord('q') or continue_num > 100:
                break
            
    env.close()
    cv2.destroyAllWindows()
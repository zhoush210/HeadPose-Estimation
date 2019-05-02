# coding=utf-8
from matplotlib import pyplot as plt
import cv2
import numpy as np
import dlib
import time
import math
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy.optimize import *

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
POINTS_NUM_LANDMARK = 68


# 手动构建7点人脸3D模型
def model_3D():
    '''
    手动构建7点人脸3D模型，以鼻尖为坐标原点
    :return: 3D模型7点矩阵 shape=(7,3)
    '''
    points_3D = np.array([
        (-225.0, -170.0, -135.0),  # 左眼左眼角
        (-75.0, -170.0, -135.0),  # 左眼右眼角，自己定的
        (75.0, -170.0, -135.0),  # 右眼左眼角，自己定的
        (225.0, -170.0, -135.0),  # 右眼右眼角
        (0.0, 0.0, 0.0),  # 鼻尖
        (-150.0, 150.0, -125.0),  # 左嘴角
        (150.0, 150.0, -125.0)  # 右嘴角
    ])
    return points_3D


# 获取最大的人脸(get_landmark7函数内调用)
def largest_face(faces):
    '''
    求最大的人脸
    :param faces:detectors检测到的多个人脸的array
    :return:最大人脸的index
    '''
    if len(faces) == 1:
        return 0
    face_areas = [(face.right() - face.left()) * (face.bottom() - face.top()) for face in faces]  # 求脸的大小
    largest_area = face_areas[0]
    largest_index = 0
    for index in range(1, len(faces)):  # 取最大的脸
        if face_areas[index] > largest_area:
            largest_index = index
            largest_area = face_areas[index]
    print("largest_face index is {} in {} faces".format(largest_index, len(faces)))
    return largest_index


# 提取68个特征点(get_landmark7函数内调用)
def get_landmarks68(img):
    '''
    用dlib提取68个特征点
    :param img: 输入图像
    :return: 68个点的坐标，shape=(68，2)
    '''
    rects = detector(img, 1)
    if len(rects) == 0:
        return -1
    return np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])


# 注释68个标记点(测试用)
def annotate_landmarks68(im,landmarks68):
    '''
    注释68个标记点
    :param im: 输入图片
    :return: 带有68个标记点的图片
    '''
    img = im.copy()
    for idx, point in enumerate(landmarks68):
        pos = (point[0, 0], point[0, 1])
        # cv2.putText(img, str(idx), pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.4,color=(0, 0, 255))
        cv2.circle(img, pos, 3, color=(0, 255, 255))
    return img


# 选取dlib检测的68点中的7个特征点
def get_landmark7(img,landmarks68):
    '''
    用dlib获取人脸7个特征点
    :param img: 输入图片
    :return: 若检测到人脸，返回:1,人脸7特征点的矩阵shape=(7,2);若未检测到人脸，返回:-1,None
    '''
    faces = detector(img, 0)  # 检测图片中的所有人脸,网上都是1，cvdlib中是0
    if len(faces) == 0:  # 没有检测到人脸
        print("ERROR: found no face")
        return -1, None,None
    largest_index = largest_face(faces)  # 取最大人脸
    face_rectangle = faces[largest_index]  # 取对应人脸框
    landmark68 = predictor(img, face_rectangle)  # dlib检测人脸特征68点
    landmark7 = np.array([  # 取出68点中所需的7个点
        (landmark68.part(36).x, landmark68.part(36).y),  # 左眼左眼角
        (landmark68.part(39).x, landmark68.part(39).y),  # 左眼右眼角
        (landmark68.part(42).x, landmark68.part(42).y),  # 右眼左眼角
        (landmark68.part(45).x, landmark68.part(45).y),  # 右眼右眼角
        (landmark68.part(30).x, landmark68.part(30).y),  # 鼻尖
        (landmark68.part(48).x, landmark68.part(48).y),  # 左嘴角
        (landmark68.part(54).x, landmark68.part(54).y)  # 右嘴角
    ], dtype="double")
    return 1, landmark7,landmark68


# 归一化点
def get_normalize(points):
    '''
    归一化点
    :param 手动构建的3D模型(points_3D) 或 dlib检测到的7个特征点(points_2D)
    :return: 归一化后的点矩阵
    '''
    center = np.sum(points, axis=0) / points.shape[0]  # 中心
    L = np.sum(np.sum((points - center) ** 2, axis=1) ** 0.5)  # 归一化系数
    normalize = (points - center) / L
    return normalize, center, L


# 求投影矩阵P
def get_P_matrix(normalize_3D, normalize_2D):
    num = normalize_2D.shape[0]
    _ = 0  # _是要舍弃的值
    if num < 3:
        return -1
    elif num == 3:
        _, r1_T = cv2.solve(normalize_3D.T, normalize_2D.T[0], _)
        _, r2_T = cv2.solve(normalize_3D.T, normalize_2D.T[1], _)
    else:
        _, r1_T = cv2.solve(normalize_3D, normalize_2D.T[0], _, cv2.DECOMP_SVD)  # 奇异值分解
        _, r2_T = cv2.solve(normalize_3D, normalize_2D.T[1], _, cv2.DECOMP_SVD)
    r1 = r1_T.T[0]
    r2 = r2_T.T[0]
    r3 = np.cross(r1, r2)
    P = np.array([r1, r2])
    return P


# 定义送入优化器的修正的目标函数
def f(K):
    '''
    修正的目标函数
    :param K: 待优化的参数,K*Z中的K
    :return: 修正的目标函数的值
    '''
    a = 0.00000001
    disparity = (normalize_2D - np.dot(normalize_3D, P.T)).T  # disparity:d=b-P*a
    objective = np.sum(disparity ** 2)  # 目标函数
    penalty = np.sum((np.ones(K.shape[0]) - K) ** 2 * normalize_3D.T[2] ** 2)  # 惩罚函数
    revised_objective = objective + a * penalty  # 修正的目标函数
    return revised_objective


# 得到K修正后的3D模型(A_K)
def get_normalize_3D_K(normalize_3D, K_opt):
    '''
    计算经K修正的3D模型
    :param normalize_3D: 归一化的3D模型
    :param K_opt: 最优化K
    :return: K修正的3D模型
    '''
    normalize_3D_K = np.array([normalize_3D.T[0], normalize_3D.T[1], normalize_3D.T[2] * K_opt])
    return normalize_3D_K.T


# 得到K修正后的最优投影矩阵
def get_P_K_opt(normalize_2D, normalize_3D_K):
    '''
    计算K修正后的最优投影矩阵
    :param normalize_2D: 归一化2D特征点
    :param normalize_3D_K: 归一化K修正3D模型
    :return:最优投影矩阵
    '''
    P_K_opt = np.dot(np.dot(normalize_2D.T, normalize_3D_K), np.linalg.pinv(np.dot(normalize_3D_K.T, normalize_3D_K)))
    return P_K_opt


# 由最优投影矩阵计算欧拉角
def get_euler_angle(P_K_opt):
    '''
    由最优投影矩阵计算欧拉角
    :param P_opt: 最优投影矩阵
    :return: 欧拉角
    '''
    theta = np.array([0, 0, 0])  # (theta x,y,z)
    r1 = P_K_opt[0]
    r2 = P_K_opt[1]
    r3 = np.cross(r1, r2)
    # print('r3=', r3)
    theta[0] = -math.atan(r3[1] / r3[2]) / math.pi * 180  # 论文和data的参考系不一致，前面加负号和data一致
    theta[1] = -math.atan(r3[0] / (r3[1] ** 2 + r3[2] ** 2) ** 0.5) / math.pi * 180
    theta[2] = -math.atan(r2[0] / r1[0]) / math.pi * 180  # 论文和data的参考系不一致，前面加负号和data一致
    return theta


# 画表示姿势的线
def draw_line(img, points_2D, P_K_opt):
    '''
    画表示姿势的线
    :param img: 输入图片
    :param points_2D: dlib提取的2D特征点
    :return: 画有姿势线的图片
    '''
    nose_tip_2D = (int(points_2D[4][0]), int(points_2D[4][1]))
    far_3D = np.array([0.0, 0.0, 1000.0])
    far_2D = np.dot(np.dot(P_K_opt, (far_3D - center_3D) / L_3D), L_2D) + center_2D
    p1 = (int(nose_tip_2D[0]), int(nose_tip_2D[1]))
    p2 = (int(far_2D[0]), int(far_2D[1]))
    # print('p1=', p1, 'p2=', p2)
    img_with_line = np.copy(img)
    cv2.line(img_with_line, p1, p2, (255, 0, 0), 2)
    return img_with_line


cap = cv2.VideoCapture(0)  # 实例化摄像头

while (cap.isOpened()):
    start_time = time.time()
    _, img = cap.read()  # 输入图片

    landmark68=get_landmarks68(img)
    ret, points_2D = get_landmark7(img,landmark68)  # dlib检测7个特征点
    points_3D = model_3D()  # 手动构建7点人脸3D模型

    if ret == 1:  # 检测到人脸7个特征点
        normalize_3D, center_3D, L_3D = get_normalize(points_3D)  # 归一化
        normalize_2D, center_2D, L_2D = get_normalize(points_2D)

        P = get_P_matrix(normalize_3D, normalize_2D)  # 投影矩阵P

        # K=np.random.random(normalize_3D.shape[0])
        K = [0, 0, 0, 0, 0, 0, 0]  # 初始化
        K_opt = scipy.optimize.fmin_cg(f, K, gtol=1e-20)  # 共轭梯度下降法

        normalize_3D_K = get_normalize_3D_K(normalize_3D, K_opt)  # K修正后的3D模型

        P_K_opt = get_P_K_opt(normalize_2D, normalize_3D_K)  # K修正后的最优投影矩阵

        theta = get_euler_angle(P_K_opt)  # 计算欧拉角

        img_with_line = draw_line(img, points_2D, P_K_opt)  # 画表示姿势的线
        img_with_68points = annotate_landmarks68(img,landmark68)

        euler_angle_str = 'Pitch:{}, Yaw:{}, Roll:{}'.format(theta[0], theta[1], theta[2])

        fps = 1 / (time.time() - start_time)

        cv2.putText(img_with_line, euler_angle_str, (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        cv2.putText(img_with_line, "FPS : " + str(int(fps)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1)
        cv2.putText(img_with_68points, "FPS : " + str(int(fps)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Pose", img_with_line)
        cv2.imshow("68points", img_with_68points)
        cv2.waitKey(1)
    else:  # 未检测到人脸特征点
        fps = 1 / (time.time() - start_time)
        cv2.putText(img, "FPS : " + str(int(fps)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Pose", img)
        cv2.waitKey(1)

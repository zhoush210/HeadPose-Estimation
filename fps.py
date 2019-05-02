# coding=utf-8
import cv2
import time

cap = cv2.VideoCapture(0)  # 实例化摄像头
while(1):
    start_time = time.time()
    _, img = cap.read()  # 输入图片
    fps = 1 / (time.time() - start_time)
    cv2.putText(img, "FPS : " + str(int(fps)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Output", img)
    cv2.waitKey(1)

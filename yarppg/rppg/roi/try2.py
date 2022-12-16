import cv2
import dlib
import numpy
import numpy as np
import matplotlib.pyplot as plt
def test2(rawimg):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")
    img = rawimg.copy()
    while True:
        # x_1, x_2, y_1, y_2 = 0, 0, 0, 0
        # mask , mask2 =None, None
        # img_end = None
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_img, 1)
        pts = []
        for face in faces:
            landmarks = predictor(gray_img, face)
            # 提取额头区域
            for i in [19, 69, 80, 24]:
                pts = [[landmarks.part(i).x, landmarks.part(i).y]] + pts
            x_1 = min(landmarks.part(69).x, landmarks.part(19).x)
            x_2 = max(landmarks.part(80).x, landmarks.part(24).x)
            y_1 = min(landmarks.part(69).y, landmarks.part(80).y)
            y_2 = max(landmarks.part(19).y, landmarks.part(24).y)
            if x_1 < 0:
                x_1 = 0
            if x_2 > img.shape[1]:
                x_2 = img.shape[1]
            if y_1 < 0:
                y_1 = 0
            if y_2 > img.shape[0]:
                y_2 = img.shape[0]
            pts = np.array(pts)
            pts = pts.reshape((-1, 1, 2))
            #cv2.polylines(img, [pts], True, (0, 255, 255))
            mask = np.zeros(img.shape, dtype=np.uint8)
            mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
            ROI = cv2.bitwise_and(mask2, img)
            img_end = ROI[y_1: y_2, x_1: x_2]
            r, g, b, a = cv2.mean(rawimg, mask)
            # rawimg和mask为同样大小的矩阵，计算rawimg中所有元素的均值，为0的地方不计算
            # rawimg = base_img
            # 获取图像四通道各自的均值（a为透明度）

        #img_end = cv2.resize(img_end, None, fx=2, fy=2)
        # cv2.namedWindow('ROI', 0)
        # cv2.resizeWindow('ROI', 560, 420)
        # cv2.imshow('ROI', img_end)
        # cv2.imshow("face", img)
    return r, g, b
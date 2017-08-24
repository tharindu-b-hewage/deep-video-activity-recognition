import scipy.io as matLoader
import cv2

label = matLoader.loadmat("/home/lorddbaelish/PycharmProjects/deep-video-activity-recognition/KerasC3D/MERL Shopping Dataset/Labels_MERL_Shopping_Dataset/1_1_label.mat")

fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))



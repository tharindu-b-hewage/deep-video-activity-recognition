import cv2
#import numpy as np
import matplotlib.pyplot as plt


print "start .."
captured = cv2.VideoCapture('/home/lorddbaelish/PycharmProjects/deep-video-activity-recognition/KerasC3D/MERL Shopping Dataset/Videos_MERL_Shopping_Dataset/1_1_crop.mp4')
print "loaded video.."
video = []
i = 0
print "start capturing.."
FPS_REDUCE_FACTOR = 3
counter = 0
while True:
    reval, img =  captured.read()
    if not reval:
        print "broke!"
        break
    if counter==0:
        resized = cv2.resize(img, (171, 128))
        cropped = resized[8:120, 30:142, :]
        video.append(cropped)
        counter += 1
    elif counter==(FPS_REDUCE_FACTOR-1):
        counter = 0
    else:
        counter += 1



print "Done formatting: video length = " + str(len(video))
len = len(video)
for i in xrange(16):
    figure = plt.figure()
    plt.imshow(video[i+len/2])
    plt.show()


import scipy
import scipy.io
import random
import numpy as np
import cv2
from pandas._window import random

mat = scipy.io.loadmat('/home/tharindu/PycharmProjects/KerasC3D/MERL Shopping Dataset/Labels_MERL_Shopping_Dataset/1_1_label.mat')

# Process each video
# Downsample every video chunck for a specific class in to 16 frames using data from label array
# Parallely create label array relevant to each video chunk
# Those would be our training data


CNN_VIDEO_LEN = 16
NUMBER_OF_RANDOM_SAMPLES_PER_CHUNK = 5

def reduce_fps(frame_array, REDUCE_FACTOR):
    output = []
    for i in xrange(0, len(frame_array)-len(frame_array)%REDUCE_FACTOR, REDUCE_FACTOR):
        output.append(frame_array[i])
    return frame_array


def generate_data(video_path, label_path):
    data_file = []
    label_file = []
    label_data = scipy.io.loadmat(label_path)
    #load video frames
    cap = cv2.VideoCapture(video_path)
    video = []
    FPS_REDUCE_FACTOR = 1
    counter = 0
    while True:
        reval, img = cap.read()
        if not reval:
            print "broke!"
            break
        if counter == 0:
            resized = cv2.resize(img, (171, 128))
            cropped = resized[8:120, 30:142, :]
            video.append(cropped)
            counter += 1
        elif counter == (FPS_REDUCE_FACTOR - 1):
            counter = 0
        else:
            counter += 1

    #for each video categories defined on labels
    for i in xrange(5):
        #Each video chunk in the same class defined as in labels
        for video_chunk in label_data['tlabs'][0][i]:
            video_segment = reduce_fps([video_chunk[0], video_chunk[1]], 3)
            length = len(video_segment)
            list = range(length)
            randomly_picked_slices = []
            while len(randomly_picked_slices) < NUMBER_OF_RANDOM_SAMPLES_PER_CHUNK:
                i = random.randint(0, length-CNN_VIDEO_LEN)
                k = list[i:i + CNN_VIDEO_LEN-1]
                if not k in randomly_picked_slices:
                    randomly_picked_slices.append(k)

            # Now we have indexes for randomly picked samples for a specific category for our selected instance
            for slice in randomly_picked_slices:
                data_file.append(video_segment[slice[0]:slice[1]])
                label_file.append(i)

import scipy
import scipy.io
import random
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


# Process each video
# Downsample every video chunck for a specific class in to 16 frames using data from label array
# Parallely create label array relevant to each video chunk
# Those would be our training data

#--- Parameters---
CNN_VIDEO_LEN = 16
NUMBER_OF_RANDOM_SAMPLES_PER_CHUNK = 2
FPS_REDUCE_SCALE = 1


def reduce_fps(frame_array, REDUCE_FACTOR):
    output = []
    for i in xrange(0, len(frame_array)-len(frame_array)%REDUCE_FACTOR, REDUCE_FACTOR):
        output.append(frame_array[i])
    return output

def generate_data(video_path, label_path):
    data_file = []
    label_file = []
    label_data = scipy.io.loadmat(label_path)

    #load video frames
    cap = cv2.VideoCapture(video_path)
    video = []
    while True:
        reval, img = cap.read()
        if not reval:
            #print "broke!"
            break
        resized = cv2.resize(img, (171, 128))
        cropped = resized[8:120, 30:142, :]
        video.append(cropped)
    #print 'Loaded video length: '+str(len(video))

    #for each video categories defined on labels
    for category_number in xrange(5):
        #Each video chunk in the same class defined as in labels
        for video_chunk in label_data['tlabs'][category_number][0]:
            #print video_chunk
            #print 'original video chunk length:'+str(video_chunk[1]-video_chunk[0])
            video_segment = reduce_fps(video[video_chunk[0]:video_chunk[1]], FPS_REDUCE_SCALE)
            length = len(video_segment)
            randomly_picked_slices = []

            if length>CNN_VIDEO_LEN:
                while len(randomly_picked_slices) < length/CNN_VIDEO_LEN:
                    i = random.randint(0, length-CNN_VIDEO_LEN)
                    k = (i, i + CNN_VIDEO_LEN-1)
                    if not k in randomly_picked_slices:
                        randomly_picked_slices.append(k)
            elif length==CNN_VIDEO_LEN :
                randomly_picked_slices.append((0, length-1))

            # Now we have indexes for randomly picked samples for a specific category for our selected instance
            for slice in randomly_picked_slices:
                data_file.append(video_segment[slice[0]:(slice[-1]+1)])
                #print 'appending length: '+str(len(video_segment[slice[0]:(slice[-1]+1)]))
                label_file.append(category_number)

    return data_file, label_file

def writeVideo(frame_array, video_file_name):
    cv2.VideoWriter(video_file_name, )


#------------ Test this file -----------
video_path = '/home/lorddbaelish/PycharmProjects/deep-video-activity-recognition/KerasC3D/MERL Shopping Dataset/Videos_MERL_Shopping_Dataset/1_1_crop.mp4'
label_path = "/home/lorddbaelish/PycharmProjects/deep-video-activity-recognition/KerasC3D/MERL Shopping Dataset/Labels_MERL_Shopping_Dataset/1_1_label.mat"

data, label = generate_data(video_path, label_path)

#for frame in data[0]:
#    plt.imshow(frame)
#    plt.show()

import scipy
import scipy.io
import random
import numpy as np
import cv2
import random
import os.path
import matplotlib.pyplot as plt


# Process each video
# Downsample every video chunck for a specific class in to 16 frames using data from label array
# Parallely create label array relevant to each video chunk
# Those would be our training data

#--- Parameters---

class MERL_Dataset:
    @staticmethod
    def writeVideo(frame_array, video_file_name, FPS_REDUCE_SCALE=1):
        #print len(frame_array)
        video_writer = cv2.VideoWriter(video_file_name, cv2.cv.CV_FOURCC(*'XVID'), 30 / FPS_REDUCE_SCALE, (112, 112), True)
        for frame in frame_array:
            video_writer.write(frame)
        video_writer.release()

    def __init__(self):
        self.CNN_VIDEO_LEN = 16
        self.NUMBER_OF_RANDOM_SAMPLES_PER_CHUNK = 2
        self.FPS_REDUCE_SCALE = 1
        self.data_array = None
        self.label_array = None

    def reduce_fps(self, frame_array, REDUCE_FACTOR):
        output = []
        for i in xrange(0, len(frame_array) - len(frame_array) % REDUCE_FACTOR, REDUCE_FACTOR):
            output.append(frame_array[i])
        return output

    def __generate_data_per_video(self, video_path, label_path, video_list, label_list):
        label_data = scipy.io.loadmat(label_path)

        # load video frames
        cap = cv2.VideoCapture(video_path)
        video = []
        while True:
            reval, img = cap.read()
            if not reval:
                # print "broke!"
                break
            resized = cv2.resize(img, (171, 128))
            cropped = resized[16:128, 55:167, :]
            video.append(cropped)
        # print 'Loaded video length: '+str(len(video))

        # for each video categories defined on labels
        for category_number in xrange(5):
            # Each video chunk in the same class defined as in labels
            for video_chunk in label_data['tlabs'][category_number][0]:
                # print video_chunk
                #print 'original video chunk length:'+str(video_chunk[1]-video_chunk[0])
                video_segment = self.reduce_fps(video[video_chunk[0]:video_chunk[1]], self.FPS_REDUCE_SCALE)
                length = len(video_segment)
                #print length
                randomly_picked_slices = []

                if length > self.CNN_VIDEO_LEN:
                    while len(randomly_picked_slices) < length / self.CNN_VIDEO_LEN:
                        i = random.randint(0, length - self.CNN_VIDEO_LEN)
                        k = (i, i + self.CNN_VIDEO_LEN - 1)
                        if not k in randomly_picked_slices:
                            randomly_picked_slices.append(k)
                elif length == self.CNN_VIDEO_LEN:
                    randomly_picked_slices.append((0, length - 1))

                # Now we have indexes for randomly picked samples for a specific category for our selected instance
                for slice in randomly_picked_slices:
                    video_list.append(video_segment[slice[0]:(slice[-1] + 1)])
                    #print 'appending length: '+str(len(video_segment[slice[0]:(slice[-1]+1)]))
                    #print '-------- Appending dim: ' + str(video_segment[slice[0]].shape)
                    label_list.append(category_number)
        #print 'Test: Generate info = ' + str(len(video_list[0]))

    def generate_numpy_array(self, video_path, label_path, leading_start_index, leading_stop_index): # video path should be the directory of the video files and same goes for the labels
        data = []
        label = []
        for leading_index in xrange(leading_start_index, leading_stop_index+1, 1): # go through first index in the video files
            for following_index in xrange(1, 100, 1): # go through second element of the video file
                video_string = video_path + "/" + str(leading_index)+"_"+str(following_index)+"_crop.mp4"
                label_string = label_path + "/" + str(leading_index)+"_"+str(following_index)+"_label.mat"
                if os.path.exists(video_string): # check whether this file exists in the directory
                    self.__generate_data_per_video(video_string, label_string, data, label)
                else:
                    break
        self.data_array = np.array(data)
        self.label_array = np.array(label)

        return data, label

    def save(self, location_path):
        if data==None:
            return False
        np.save(location_path+"MERL_shopping_data", data)
        np.save(location_path+"MERL_shopping_label", label)
        return True



#------------ Test this file -----------
video_path = '/home/lorddbaelish/PycharmProjects/deep-video-activity-recognition/KerasC3D/MERL Shopping Dataset/Videos_MERL_Shopping_Dataset'
label_path = "/home/lorddbaelish/PycharmProjects/deep-video-activity-recognition/KerasC3D/MERL Shopping Dataset/Labels_MERL_Shopping_Dataset"

dataset = MERL_Dataset()
data, label = dataset.generate_numpy_array(video_path, label_path, 1, 2)
dataset.save("")
print 'done generating...'
while True:
    response = raw_input("Enter File: ")
    if response==-1:
        break
    print 'Category: '+str(label[int(response)])
    MERL_Dataset.writeVideo(data[int(response)], '/home/lorddbaelish/PycharmProjects/deep-video-activity-recognition/KerasC3D/test.avi')


print 'done writing....'
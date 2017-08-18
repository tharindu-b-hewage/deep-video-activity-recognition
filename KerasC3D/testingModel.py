#import os
#os.environ['THEANO_FLAGS'] = "device=gpu0"
from keras.models import model_from_json
import cv2
import numpy as np
import matplotlib.pyplot as plt


model =  model_from_json(open('sports1M_model.jason', 'r').read())
model.load_weights('sports1M_weights.h5')
model.compile(loss='mean_squared_error', optimizer='sgd')
print "****model prepared.."
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

print('Total labels: {}'.format(len(labels)))

cap = cv2.VideoCapture('video4.mp4')

vid=[]

while True:
    ret, img = cap.read()
    if not ret:
        break
    vid.append(cv2.resize(img, (171, 128)))
original_video = vid
vid = np.array(vid, dtype=np.float32)
#plt.imshow(vid[2000]/256)
#plt.show()

mean_cube = np.load('train01_16_128_171_mean.npy') ## Check whether mean is float32
mean_cube = mean_cube[:, :, 8:120, 30:142]
FRAME_START = 2000
X = vid[FRAME_START:FRAME_START+16, 8:120, 30:142, :].transpose((3, 0, 1, 2))
X = X - mean_cube
print "****video loaded.."
output = model.predict_on_batch(np.array([X]))
print "****predicted."
#print output
#plt.plot(output[0])
#plt.show()


print('Position of maximum probability: {}'.format(output[0].argmax()))
print('Maximum probability: {:.5f}'.format(max(output[0])))
print('Corresponding label: {}'.format(labels[output[0].argmax()]))

# sort top five predictions from softmax output
top_inds = output[0].argsort()[::-1][:5]  # reverse sort and take five largest items
for i in top_inds: print('{:.5f} {}'.format(output[0][i], labels[i]))
print "\n\n Playing Video in a continuous loop..."

def playClip(frames):
    for frame in frames:
        plt.imshow(frame)
        plt.pause(0.05)

playClip(original_video[FRAME_START:FRAME_START+16])
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Flatten
#from load_MERL_dataset import MERL_Dataset
import numpy as np
import sys
import os.path

#model definition
model =  model_from_json(open('sports1M_model.jason', 'r').read())
model.load_weights('sports1M_weights.h5')
for i in xrange(len(model.layers)-5): # Freeze first 15 layers
    model.layers[i].trainable = False
model.pop()
model.add(Dense(5, activation='softmax', name='fc8_MERL'))
model.compile(loss='mean_squared_error', optimizer='sgd')
print model.summary()

#load mean of sports1M set. Because we use trained layers from sports1M, applying same format of data does make sense
mean_cube = np.load('train01_16_128_171_mean.npy') ## Check whether mean is float32
mean_cube = mean_cube[:, :, 8:120, 30:142]

print 'mean cube prepared..'

#load saved numpy data of MERL dataset
data = np.load('MERL_numpy/MERL_shopping_data_1.npy')
label = np.load('MERL_numpy/MERL_shopping_label_1.npy')

print 'merl data loaded....'

#reduce mean from data : Shape after this = (number of samples, 3, 16, 112(height), 112(width))
mean_reduced_data = np.array([(video_clip.transpose(3, 0, 1, 2)-mean_cube) for video_clip in data])

print 'mean reduced.. : num of items = ' + str(mean_reduced_data.shape)

#train
train_history = model.fit(mean_reduced_data, label, batch_size=10, nb_epoch=1, validation_split=0.1)

print 'training done.'

#Save model
# serialize model to JSON
model_json = model.to_json()
for file_number in xrange(sys.maxint):
    if not os.path.exists('Saved_Instances/'+str(file_number)+'model.json'):
        with open('Saved_Instances/'+str(file_number)+'model.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights('Saved_Instances/'+str(file_number)+'model.h5')
        print("Saved model to disk")
        break

#Test loop
def test_against_data(model, data, label):
    predicted_results = model.predict(mean_reduced_data, batch_size=20)
    comparison = [predicted_results[i].argmax()==label[i].argmax() for i in xrange(data.shape[0])]
    sucess_count = comparison.count(True)
    total_size = data.shape[0]
    print 'Successive guesses: ' + str(sucess_count) + ' out of ' + str(total_size) + ' and Success percentage: ' + str(float(sucess_count/total_size))

test_against_data(model, mean_reduced_data, label)

#while True:
   # video_index = int(raw_input('Enter video number: '))
    #MERL_Dataset.writeVideo(data[0], 'evaluate_test.avi')
    #print 'Target output = '+ str(label[video_index])+' | predicted output = '+str(model.predict(np.array([mean_reduced_data[video_index]]), batch_size=1))
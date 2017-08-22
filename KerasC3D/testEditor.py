import scipy.io as matLoader

label = matLoader.loadmat("/home/lorddbaelish/PycharmProjects/deep-video-activity-recognition/KerasC3D/MERL Shopping Dataset/Labels_MERL_Shopping_Dataset/1_1_label.mat")

print label['tlabs'][1][0]




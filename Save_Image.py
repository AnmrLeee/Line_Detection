import numpy as np
import pickle
import cv2

train_images = pickle.load(open("full_CNN_labels.p", "rb"))
# print(train_images[1])

train_images = np.array(train_images)
print(train_images.shape[0])


trainSet = np.zeros((80, 160))
trainSet2 = np.zeros((1, 12800))

for i in range(train_images.shape[0]):
    trainSet2 = train_images[i][0:]
    trainSet = trainSet2.reshape(80, 160)

    path = '/home/an/PycharmProjects/Line_detection/label_image/'+str(i)+'.png'
    cv2.imwrite(path, trainSet)
    # cv2.namedWindow('trainSet', 0)
    # cv2.imshow('trainSet', trainSet)

if cv2.waitKey() == '27':
    cv2.destroyAllWindows()

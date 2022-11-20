import pandas as pd
import numpy as np
from skimage.feature import hog
import cv2
import os
import scipy
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import neighbors
from keras.utils.np_utils import to_categorical



DA = os.listdir("C:\datasets\data\project\daily activity")
fall = os.listdir("C:\datasets\data\project\FA cam1")
filepath="C:\datasets\data\project\daily activity"
filepath2="C:\datasets\data\project\FA cam1"




images = []
label = []
y = []


#loading daily activity images
for i in DA:
    image = scipy.misc.imread( os.path.join(filepath,i))
    images.append(image)
    label.append(0) #for daily activity images
    y.append(i)


#loading fall activity images
for i in fall:
    image = scipy.misc.imread( os.path.join(filepath2,i))
    images.append(image)
    label.append(1) #for fall activity images
    y.append(i)



#extracting features 
df = pd.DataFrame()
for i in range(0,118):
    images[i] = cv2.resize(np.array(images[i]),(400,400))
    fd, hog_image = hog(images[i], orientations=16, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    df[ y[i] ] = fd

# 36864 features extracted for each image - 24*24*(16*4)
# ((400/16)-1)*((400/16)-1)*TotalNumberOfOrientations(bins)     


df = df.T
df = df.values.astype('float32')
df = preprocessing.scale(df)


#splitting training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(df, label, test_size=0.2)


ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)

#defining SVM model
model = SVC(kernel='linear')
model.fit(Xtrain, ytrain)


#predicting on test split using SVM model
scores = model.predict(Xtest)
score = pd.DataFrame(scores)
ytest = pd.DataFrame(ytest)
error = pd.concat([ytest,score],axis=1)
error['error'] = error.iloc[:,0] - error.iloc[:,1]
efficiency_SVM = error.ix[error['error']==0,:].shape[0]/error.shape[0]



#defining kNN model
model = neighbors.KNeighborsRegressor(n_neighbors = 3)
model.fit(Xtrain, ytrain)  #fit the model
scores=model.predict(Xtest) #make prediction on test set
score = pd.DataFrame(scores)
ytest = pd.DataFrame(ytest)
error = pd.concat([ytest,score],axis=1)
error['error'] = error.iloc[:,0] - error.iloc[:,1]
efficiency_kNN = error.ix[error['error']==0,:].shape[0]/error.shape[0]


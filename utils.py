
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
import subprocess
import sys
# In[3]:

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from os.path import join as opj
from matplotlib import pyplot as plt





#Import Keras.
from matplotlib import pyplot
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import Convolution2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.optimizers import rmsprop
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.datasets import cifar10
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg19 import VGG19
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input 

#Data Aug for multi-input
from keras.preprocessing.image import ImageDataGenerator
import pickle


def submit_and_shutdown(phase, exp_name, train_preds, val_preds,test_preds, train_log_loss,valid_log_loss,shutdown=True):

		lb  = 0.0
		try:
			submission_output = subprocess.check_output(["kg", "submit",'subm/{}.csv'.format(exp_name)])
			lb = float(submission_output)
		
		except:
			print (submission_output)
		

		update_results_h5("ph1",exp_name,train_preds,val_preds,test_preds,train_log_loss,valid_log_loss,LB_score=lb)

		if shutdown:
			subprocess.call(["sudo", "shutdown", "now"])


def update_results_h5(phase, exp_name, train_preds, val_preds,test_preds, train_log_loss,valid_log_loss, LB_score=0.0 ):
        store = pd.HDFStore('data/results.h5')
        store.append("/{}/train/{}".format(phase,exp_name),train_preds )
        store.append("/{}/valid/{}".format(phase,exp_name),val_preds)
        store.append("/{}/test/{}".format(phase,exp_name),test_preds)  
                     
        store.append("/summary",pd.DataFrame(data={"phase":[phase],
                                                   "exp":[exp_name],
                                                   "train_log_loss":[train_log_loss],
                                                   "val_log_loss":[valid_log_loss], 
                                                   "LB":[LB_score] }) )  

        store.close()
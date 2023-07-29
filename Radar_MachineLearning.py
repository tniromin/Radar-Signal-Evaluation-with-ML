
#Data Set Information:

#This radar data was collected by a system in Goose Bay, Labrador. This system consists of a phased array of 16 high-frequency antennas with a total transmitted power on the order of 6.4 kilowatts. See the paper for more details. The targets were free electrons in the ionosphere. "Good" radar returns are those showing evidence of some type of structure in the ionosphere. "Bad" returns are those that do not; their signals pass through the ionosphere.
#Received signals were processed using an autocorrelation function whose arguments are the time of a pulse and the pulse number. There were 17 pulse numbers for the Goose Bay system. Instances in this databse are described by 2 attributes per pulse number, corresponding to the complex values returned by the function resulting from the complex electromagnetic signal.

#Attribute Information:

#All 34 are continuous
#The 35th attribute is either "good" or "bad" according to the definition summarized above. This is a binary classification task.

#Relevant Papers:
#Sigillito, V. G., Wing, S. P., Hutton, L. V., & Baker, K. B. (1989). Classification of radar returns from the ionosphere using neural networks. Johns Hopkins APL Technical Digest, 10, 262-266. 

#Url =https://www.kaggle.com/datasets/prashant111/ionosphere/code

# Training data repository https://github.com/selva86/datasets >> Star
import pandas as pd
url="https://raw.githubusercontent.com/selva86/datasets/master/Ionosphere.csv"
df=pd.read_csv(url)

#Import Keras Library for Nueron models

from keras.models import Sequential , load_model
from keras.layers import Dense ,  Dropout

# declare X,Y
x=df.drop(['Class'],axis=1)
y=df['Class']

#add splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



model = Sequential()
model.add (Dense(units=10, activation ='relu',input_dim=len(x_train.columns))) # input dimension
model.add (Dense(units=10,activation='relu'))
model.add(Dense(units=1,activation='sigmoid')) #Output sigmoid is a probability based output
model.summary()

#compiling model
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics='accuracy');

#Training data
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=200,batch_size=32); 
#Epochs => how many times the neural network should be trained with the data
#9  = training data / batchsize = No. of iterations of training data

#save the model
model.save('weights_Name.h5'); # Give relevant name for the dataset

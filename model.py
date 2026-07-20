import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

#data load
with open("array/images.p","rb") as f:
    x=pickle.load(f)                        #Input
with open("array/labels.p","rb") as f:
    y=pickle.load(f)                        #Output


#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2)

#to categorical
num_classes = len(np.unique(y))

y_train=to_categorical(y_train,num_classes)
y_test=to_categorical(y_test,num_classes)

#cnn model
model=Sequential()
#ConvLayer and Pooling layer 
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

#Fllaten
model.add(Flatten())

#Fully connected layer
#Hidden
model.add(Dense(128,activation='relu'))
#Dropout
model.add(Dropout(0.5))
#Output
model.add(Dense(num_classes,activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

#Early stopping
early_stop = EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True)


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20,batch_size=32,callbacks=[early_stop])

model.save("mask.h5")
print("model saved")
 

















import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

model=load_model("mask.h5")

classes =os.listdir(os.path.join(os.getcwd(),'data'))

img_path="OIP.jpg"
img=cv2.imread(img_path)
if img is None:
    print("image is not found")
    exit()
    
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis("off")
plt.show()

#img processing
img=cv2.resize(img,(128,128))
img=img/255
img=np.reshape(img,(1,128,128,3))  

#prediction
pred= model.predict(img)
class_index = np.argmax(pred)
con=np.max(pred)

print("Prediction :",classes[class_index])
print("Confidence:",float(con)*100,'%')








































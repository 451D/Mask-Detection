import cv2
import os
import pickle
import numpy as np

#dataset folder
dataset_dir = os.path.join(os.getcwd(),'data')    
#output folder
data_dir = os.path.join(os.getcwd(),'array')        
#2step verification
os.makedirs(data_dir,exist_ok=True)                

img_data=[]  
labels=[]

classes=os.listdir(dataset_dir)                    

label_map={cls:idx for idx,cls in enumerate(classes)}
print("label_map:",label_map)

for cls in classes:
    class_path=os.path.join(dataset_dir,cls)
    
    if not os.path.isdir(class_path):
        continue
    for file in os.listdir(class_path):
        img_path = os.path.join(class_path,file)
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        #preprocess
        image=cv2.resize(image,(128,128))
        image=image/255
        
        img_data.append(image)
        labels.append(label_map[cls])
        
#convert2Array
img_data=np.array(img_data,dtype="float32")      
labels=np.array(labels)

print("images shape:",img_data.shape)
print("label shape:",labels.shape)

#save pickle in folder
with open(os.path.join(data_dir,"images.p"),'wb') as f:
    pickle.dump(img_data,f)
with open(os.path.join(data_dir,"labels.p"),'wb') as f:
    pickle.dump(labels,f)

print("dataset process and save")



















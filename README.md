# Mask-Detection
Detect if the person in the image has a mask on or not.

# Adding Dataset
Add the requried dataset into the data folder according to the instruction provided in the readme text file 

Dataset link: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

#  Mask Detection CNN

##  Overview
This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to classify face images for **mask detection**.  
The model is trained on a dataset of labeled images and outputs predictions across multiple classes (e.g., *mask* vs *no mask*).

---

##  Workflow
1. **Data Loading**
   - Images and labels are loaded from serialized `.p` files (`images.p`, `labels.p`) using `pickle`.
   - Labels are one‑hot encoded with `to_categorical`.

2. **Train/Test Split**
   - Dataset split into 80% training and 20% testing using `train_test_split`.

3. **Model Architecture**
   - Input: `(128, 128, 3)` RGB images.
   - Convolutional layers with increasing filters: 32 → 64 → 128 → 256.
   - MaxPooling layers for downsampling.
   - Flatten layer to convert feature maps into a vector.
   - Dense hidden layer (128 units, ReLU).
   - Dropout (0.5) for regularization.
   - Output layer with `softmax` activation for multi‑class classification.

4. **Compilation & Training**
   - Optimizer: Adam (`learning_rate=0.001`).
   - Loss: `categorical_crossentropy`.
   - Metric: Accuracy.
   - EarlyStopping callback prevents overfitting (patience=2, restore best weights).
   - Training for up to 20 epochs, batch size 32.

5. **Model Saving**
   - Final trained model saved as `mask.h5`.


##  Getting Started

### Requirements
Make sure you have the following installed:

- Python 3.8+ (Recommended: Python 3.12)
- TensorFlow / Keras
- NumPy
- scikit‑learn
- OpenCV
- Matplotlib
- pickle (built‑in with Python)

### File runing sequence
- First: array.py (Convert dataset to array and store in pickle formate)
- Second: model.py (Fit and save the model in 'mask.h5' )
- Third: test.py (For testing the model)

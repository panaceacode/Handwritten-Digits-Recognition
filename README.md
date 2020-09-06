# Handwritten-Digits-Recognition
Handwritten digits recognition using OpenCV, keras and python


Dependencies
1. cv2
2. numpy
3. keras
4. keras.models (Sequential)
5. keras.layers (Dense, Dropout, Flatten, Con2D, MaxPooling2D)


Contents
This zip file contains the following files:
1. train.py
2. test.py
3. digit_recognition.json (CNN’s structure is saved in this file)
4. weights_records.h5 (Model’s weights are saved in this file)


Usage
For train.py file, the first part is data processing. CNN’s structure is at the bottom of this file.
For test.py file (where we apply the model to recognize handwritten digits):
1. Add test file to this fold, use the digit_recognition(x) to generate the classification results.
   sample: y_pre = digit_recognition( ‘filename.npy’ )
	   then, y_pre will be saved in a file called ‘results.npy‘
	   Also, you can print(y_pre)
2. We provide accuracy(y_pre, label_file) function to calculate the overall accuracy if have ‘label_file.npy’ file.
   sample: accuracy(y_pre, ‘label_file.npy’), y_pre is the results and label_file.npy should be added to this fold

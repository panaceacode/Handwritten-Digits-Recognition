import cv2
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Load training data and corresponding labels
X_train_raw = np.load('X_train.npy')
y_train_raw = np.load('y_train.npy')

# Shuffle the training set before we train the model
index = np.arange(1200)
np.random.shuffle(index)
X_train = X_train_raw[index, :, :]
y_train = y_train_raw[index]


# Find the upper edge of the digital region
# X is 2D image, threshold stands for the standard of distinguishing the digit and background
# cnt_up is the upper limit of count of success, init is the reserved rows besides digital region
def find_upedge(X, threshold, cnt_up, init):
    cnt = 0  # Count of success
    a = 0  # The up edge
    flag = 0  # flag of success
    for k in range(0, len(X)):  # Scan each row of the image
        xmin = np.argmin(X[k])  # Ax pixel of this row
        xmax = np.argmax(X[k])  # Min pixel of this row
        if X[k, xmax] - X[k, xmin] > threshold: # If the difference is bigger than threhold, digit or noise may still exist in this row
            flag = 0  # Reset the flag of success
            for i in range(0, len(X[0])-6): # Scan every pixel of this row
                if abs(X[k, i] - X[k, +5]) > threshold and abs(X[k, i+1] - X[k, i+6]) > threshold: # Check two consecutive gaps so that the possiblitiy of the noise be ruled out
                    flag = 1  # This row maybe digit,set success flag to 1
                    break
            if flag == 1:  # Find digit successfully
                if cnt == 0:
                    cnt = cnt + 1  # Count of success
                    a = k  # Record the initial row of success
                elif cnt > cnt_up:  # Satisfy the upper limit of count
                    return max(a-init, 0)  # Return the initial row containing the digit with minus reserved rows. If negative, return the first row
                else:  # Not satisfy the upper limit of count
                    cnt = cnt+1  # Count of success
            else:  # Find digit not successfully
                cnt = 0  # Reset
                a = 0  # Reset
        else:
            cnt = 0  # Reset
            a = 0  # Reset
    return 0  # Not find upper edge of digit region, return the first row


# Find the lower edge of the digital region
def find_downedge(X, threshold, cnt_up, init):
    cnt = 0
    a = 0
    flag = 0
    for k in range(len(X)-1, -1, -1): # Scan each row of the image starting at the last row
        xmin = np.argmin(X[k])
        xmax = np.argmax(X[k])
        if X[k, xmax] - X[k, xmin] > threshold:
            flag = 0
            for i in range(0, len(X[0])-6):
                if abs(X[k, i] - X[k, i+5]) > threshold and abs(X[k, i+1]-X[k, i+6]) > threshold:
                    flag = 1
                    break
            if flag == 1:
                if cnt == 0:
                    cnt = cnt+1
                    a = k
                elif cnt > cnt_up:
                    return min(a+init, len(X)-1) # Return the initial row containing the digit adding reserved rows. If out of range, return the last row
                else:
                    cnt = cnt + 1
            else:
                cnt = 0
                a = 0
        else:
            cnt = 0
            a = 0
    return len(X) - 1  # Not find lower edge of digit region, return the last row


# find the left edge of the digital region
def find_leftedge(X, threshold, cnt_up, init):
    cnt = 0
    a = 0
    flag = 0
    for k in range(0, len(X[0])):  # Scan each column of the image
        xmin = np.argmin(X[:, k])
        xmax = np.argmax(X[:, k])
        if X[xmax, k] - X[xmin, k] > threshold:
            flag = 0
            for i in range(0, len(X)-6):
                if abs(X[i, k] - X[i+5, k]) > threshold and abs(X[i+1, k] - X[i+6, k]) > threshold:
                    flag = 1
                    break
            if flag == 1:
                if cnt == 0:
                    cnt = cnt+1
                    a = k
                elif cnt > cnt_up:
                    return max(a-init,0)
                else:
                    cnt = cnt+1
            else:
                cnt = 0
                a = 0
        else:
            cnt = 0
            a = 0
    return 0  # not find left edge of digit region, return the first column


def find_rightedge(X, threshold, cnt_up, init):
    cnt = 0
    a = 0
    flag = 0
    for k in range(len(X[0])-1, -1, -1):  # scan each column of the image starting at the last column
        xmin = np.argmin(X[:, k])
        xmax = np.argmax(X[:, k])
        if X[xmax, k] - X[xmin, k] > threshold:
            flag = 0
            for i in range(0, len(X) - 6):
                if abs(X[i, k] - X[i+5, k]) > threshold and abs(X[i+1, k] - X[i+6, k]) > threshold:
                    flag = 1
                    break
            if flag == 1:
                if cnt == 0:
                    cnt = cnt+1
                    a = k
                elif cnt > cnt_up:
                    return min(a+init, len(X[0])-1)
                else:
                    cnt = cnt + 1
            else:
                cnt = 0
                a = 0
        else:
            cnt = 0
            a = 0
    return len(X[0])-1  # not find right edge of digit region, return the last column


# Preprocess the training set
N = len(X_train) # Number of samples
height = 32 # Height of resized image
width = 32 # Width of resized image

# Extract the digital region and resize
X_train_trans = np.zeros((N, height, width)) # Create an empty dataset
for i in range(0, len(X_train)):
    y = X_train[i]
    y = (y - np.min(y)) / (np.max(y) - np.min(y))  # Normalize
    a = find_upedge(y, 0.3, 2, 5)  # Reserve 5 rows up to upper edge and lower edge
    b = find_downedge(y, 0.3, 4, 5)
    c = find_leftedge(y[a:b, :], 0.3, 2, 20)  # Reserve 20 columns up to left edge and right edge
    d = find_rightedge(y[a:b, :], 0.3, 2, 20)
    resized_y = cv2.resize(y[a:b, c:d], (width, height))  # Extract the digital region and resize it
    X_train_trans[i] = resized_y  # Copy resized image into new dataset

X_train4D = X_train_trans.reshape(X_train_trans.shape[0], height, width, 1).astype('float32')
y_train_one_hot = keras.utils.to_categorical(y_train, 10)

# Previously training process should split X_train4D and y_train_one_hot into training set and testing set

# We apply CNN to implement our project, 2 Conv2D layers, 2 MaxPooling2D layers and 2 Dense layers with dropout
# activation function of hidden layer is relu while output layer uses softmax in order to output the probabilities of
# the 10 classes.

# Training epochs and batch_size of our CNN model
epochs = 50
batch_size = 120

model = Sequential()
model.add(Conv2D(filters=16,
                 kernel_size=(3, 3),
                 padding='same',
                 input_shape=(32, 32, 1),
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=36,
                 kernel_size=(8, 8),
                 padding='same',
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_history=model.fit(x=X_train4D,
                        y=y_train_one_hot,
                        validation_split=0.2,
                        epochs=epochs,
                        batch_size=batch_size, verbose=0)

# Save the model
json_string = model.to_json()
open('digit_recognition.json', 'w').write(json_string)
# Save the weights
model.save_weights('weights_records.h5', overwrite=True)

import pandas as pd
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import joblib

# Function to load and preprocess images
def load_image(image_path, img_width, img_height):
    img = Image.open(image_path)
    img = img.resize((img_width, img_height))  # Resize
    img_array = np.array(img) / 255.0  # Normalize
    return img_array

df = pd.read_csv("artinnovate_dataset.csv")
df["price"] = df["price"].astype('int32')

train_df, test_df = train_test_split(df, test_size=0.2)

img_width, img_height = 150, 150

train_images = []
train_labels = []

for index, row in train_df.iterrows():
    image_path = os.path.join('/home/cdsw/artinnovate_dataset', row['Title'])
    img_array = load_image(image_path, img_width, img_height)
    train_images.append(img_array)
    train_labels.append(row['price'])

train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = []
test_labels = []

for index, row in test_df.iterrows():
    image_path = os.path.join('/home/cdsw/artinnovate_dataset', row['Title'])
    img_array = load_image(image_path, img_width, img_height)
    test_images.append(img_array)
    test_labels.append(row['price'])

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# CNN model (Determined after Experiments)
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)

# Train the model
model.fit(train_images, train_labels, 
          validation_data=(test_images, test_labels),
          epochs=30,
          batch_size=32,
          callbacks=[checkpointer])

test_mae = model.evaluate(test_images, test_labels, verbose=0)
print('Test MAE:', test_mae)

joblib.dump(model, 'CNN-Beta.joblib')

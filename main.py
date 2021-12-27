import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np

def process_image(filepath, *, color=None):
    image = cv2.imread(filepath)
    if color != None:
        image = cv2.cvtColor(image, color)
    imageRescale = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    imageRescale = imageRescale.astype(np.float64)
    imageRescale /= 255
    return imageRescale


def get_file_categories():
    train_label_mapping = {}
    for number, folder in enumerate(os.listdir(os.path.join(os.getcwd(), 'train_img'))):
        train_label_mapping[number] = folder
    return train_label_mapping


def get_file():
    train_image = []
    train_filename = []
    for number, folder in enumerate(os.listdir(os.path.join(os.getcwd(), 'train_img'))):
        for image_name in os.listdir(os.path.join(os.getcwd(), 'train_img', folder)):
            # append folder name as label
            train_filename.append(number)

            # append image to list as data
            image = process_image(os.path.join(os.getcwd(), 'train_img', folder, image_name))
            train_image.append(image)
    return (train_image, train_filename)


def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(3))

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def train_model(model, train_image, train_filename):
    checkpoint_path = "checkpoint/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,
                                                             verbose=1)

    history = model.fit(train_image, train_filename, epochs=15, callbacks=[checkpoint_callback])


def test_model(model, mapping):
    images = []
    results = []

    images_RGB = []
    for images_path in os.listdir(os.path.join(os.getcwd(), 'test_img')):
        # get image
        image = process_image(os.path.join(os.getcwd(), 'test_img', images_path))
        images.append(image)
        image = np.array(image)

        # predict the image
        result = model.predict(np.expand_dims(image, axis=0))
        result = np.argmax(result[0])
        results.append(mapping[result])

        # get rgb image to display for
        imageRGB = process_image(os.path.join(os.getcwd(), 'test_img', images_path), color=cv2.COLOR_BGR2RGB)
        images_RGB.append(imageRGB)

    plt.figure(figsize=(12, 12))
    for i in range(len(images_RGB)):
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images_RGB[i])
        plt.xlabel(results[i])
    plt.show()


# Create data phase
train_image, train_filename = get_file()
train_image = np.array(train_image)
train_filename = np.array(train_filename)
print(train_filename, train_image)

# Create model phase
model = create_model()

# Training model phase
train_model(model, train_image, train_filename)
model.save('model')
print("Model saved!")

# Testing Phase
model.load_weights(os.path.join(os.getcwd(), 'checkpoint/cp.ckpt'))
model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'model'))
test_model(model, get_file_categories())

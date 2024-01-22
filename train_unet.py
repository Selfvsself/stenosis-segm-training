import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from skimage.io import imread
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.models import Model

IMG_HEIGHT = 320
IMG_WIDTH = 320
IMG_CHANNELS = 3

NUM_TEST_IMAGES = 10

train_dir_images = "dataset/train/images"
train_dir_labels = "dataset/train/labels"
train_dir_labels2 = "dataset/train/labels2"
train_dir_labels3 = "dataset/train/labels3"
train_dir_labels4 = "dataset/train/labels4"
val_dir_images = "dataset/val/images"
val_dir_labels = "dataset/val/labels"
val_dir_labels2 = "dataset/val/labels2"
val_dir_labels3 = "dataset/val/labels3"
val_dir_labels4 = "dataset/val/labels4"
train_labels_names = os.listdir(train_dir_labels)
val_labels_names = os.listdir(val_dir_labels)
X_train = np.zeros((len(train_labels_names), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
Y_train = np.zeros((len(train_labels_names), IMG_HEIGHT, IMG_WIDTH, 4))
X_test = np.zeros((len(val_labels_names), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
Y_test = np.zeros((len(val_labels_names), IMG_HEIGHT, IMG_WIDTH, 4))


def train():
    for i, mask in enumerate(train_labels_names):
        msk = imread(train_dir_labels + "/" + mask, as_gray=True)
        msk2 = imread(train_dir_labels2 + "/" + mask, as_gray=True)
        msk3 = imread(train_dir_labels3 + "/" + mask, as_gray=True)
        msk4 = imread(train_dir_labels4 + "/" + mask, as_gray=True)
        msk_all = np.stack((msk, msk2, msk3, msk4), axis=-1)
        Y_train[i] = msk_all
        img = imread(train_dir_images + "/" + mask.replace("L1", "C1"))
        img = img / 255.
        X_train[i] = img

    print("Image data shape is: ", X_train.shape)
    print("Mask data shape is: ", Y_train.shape)

    for i, mask in enumerate(val_labels_names):
        msk = imread(val_dir_labels + "/" + mask, as_gray=True)
        msk2 = imread(val_dir_labels2 + "/" + mask, as_gray=True)
        msk3 = imread(val_dir_labels3 + "/" + mask, as_gray=True)
        msk4 = imread(val_dir_labels4 + "/" + mask, as_gray=True)
        msk_all = np.stack((msk, msk2, msk3, msk4), axis=-1)
        Y_test[i] = msk_all
        img = imread(val_dir_images + "/" + mask.replace("L1", "C1"))
        img = img / 255.
        X_test[i] = img

    print("Image data shape is: ", X_test.shape)
    print("Mask data shape is: ", Y_test.shape)

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(4, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[tensorflow.keras.metrics.MeanIoU(num_classes=2)])

    print(model.summary())

    dir_path = "models/unet/02/"
    os.makedirs(dir_path)
    file_name = "model_best.{epoch:02d}-{loss:.4f}.h5"
    filepath = dir_path + file_name

    earlystopper = EarlyStopping(patience=5, verbose=1)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')

    callbacks_list = [earlystopper, checkpoint]

    history = model.fit(X_train, Y_train, validation_split=0.25, batch_size=16, epochs=200,
                        callbacks=callbacks_list)

    model.save(dir_path + 'model_last.h5')

    plt.figure(figsize=(15, 15))
    plt.plot(history.history['loss'],
             label='Показатель ошибок на обучающем наборе')
    plt.plot(history.history['val_loss'],
             label='Показатель ошибок на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Показатель ошибок')
    plt.legend()
    plt.savefig(dir_path + 'loss.png')


if __name__ == '__main__':
    train()

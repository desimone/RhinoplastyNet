from __future__ import absolute_import, division, print_function

from os import environ, getcwd
from os.path import join

import numpy as np
import pandas as pd
import sklearn as skl
import csv
import keras
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.metrics import binary_accuracy, binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

pd.set_option('display.max_rows', 20)
pd.set_option('precision', 4)
np.set_printoptions(precision=4)

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Shut up tensorflow!
print("tf : {}".format(tf.__version__))
print("keras : {}".format(keras.__version__))
print("numpy : {}".format(np.__version__))
print("pandas : {}".format(pd.__version__))
print("sklearn : {}".format(skl.__version__))

# Hyper-parameters / Globals
BATCH_SIZE = 512  # tweak to your GPUs capacity
IMG_HEIGHT = 224  # ResNetInceptionv2 & Xception like 299, ResNet50/VGG/Inception 224, NASM 331
IMG_WIDTH = IMG_HEIGHT
CHANNELS = 3
DIMS = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)  # blame theano
MODEL_TO_EVAL = './models/MobileNetV2.hdf5'
EVAL_DIR = 'data/sorted2/test'

eval_datagen = ImageDataGenerator(rescale=1. / 255)
eval_generator = eval_datagen.flow_from_directory(
    EVAL_DIR,
    class_mode='binary',
    shuffle=False,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)
n_samples = eval_generator.samples
filenames = eval_generator.filenames
classes = eval_generator.classes
base_model = MobileNetV2(input_shape=DIMS, weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)  # comment for RESNET
x = Dense(1, activation='sigmoid', name='predictions')(x)
model = Model(inputs=base_model.input, outputs=x)
model.load_weights(MODEL_TO_EVAL)
model.compile(optimizer=Adam(lr=1e-3), loss=binary_crossentropy, metrics=['binary_accuracy'])

# evaluate
#
# score, acc = model.evaluate_generator(
#     eval_generator,
#     n_samples / BATCH_SIZE,
#     workers=8,
# )
# print(model.metrics_names)
# print('==> Metrics with eval')
# print("loss :{:0.4f} \t Accuracy:{:0.4f}".format(score, acc))

# predict
#
predictions = model.predict_generator(
    eval_generator,
    n_samples / BATCH_SIZE,
    workers=8,
)
rows = zip(filenames, predictions, classes)
with open("eval.csv", "w") as f:
    fieldnames = ['filepath', 'correct', 'y_pred', 'y_actual', 'before_prob', 'after_prob']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        filepath = row[0].rsplit('-', 1)[1]
        raw_prob = row[1][0]
        y_actual = row[2]
        after_prob = 1 - raw_prob
        before_prob = raw_prob
        y_pred = 1 if raw_prob > .5 else 0
        is_correct = y_pred == y_actual
        writer.writerow({
            'filepath': filepath,
            'correct': is_correct,
            'y_pred': y_pred,
            'y_actual': y_actual,
            'before_prob': before_prob,
            'after_prob': after_prob,
        })

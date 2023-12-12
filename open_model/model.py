# Copyright 2023 Rowen Horbach, Eline R. de Groot, Jeroen Dudink, Ronald Poppe.

# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.

# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with 
# this program. If not, see <https://www.gnu.org/licenses/>.

import sys
import os
import tensorflow as tf
import cv2
import numpy as np
from keras.models import model_from_json

import pathlib
sys.path.append(pathlib.Path(__file__).parent.resolve())
import settings as settings
import src.base_model
import src.model_helper as helper

class OpenModel(src.base_model.BaseModel):
    learning_rate = 0.01
    iterations = 1 # Number of epoch changes
    interval = 5 # 100 # Number of epochs per learning rate
    patience = 10 # Patience for early stopping
    default_test_ratio = 0.2 # Test ratio used if no test items are defined otherwise
    default_val_ratio = 0.15 # Val ratio used if no val items are defined otherwise

    def __init__(self):
        super().__init__(str(pathlib.Path(__file__).parent.resolve()).replace("\\", "/") + "/")

    def load_model(self):
        f = open(self.read_dir + "model.json", "r")
        model = model_from_json(f.read())
        model.load_weights(self.read_dir + "model")
        return model

    def get_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(6,kernel_size=(5, 5),strides=2,activation='relu',input_shape=(56, 56, 3),padding='valid'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
        model.add(tf.keras.layers.BatchNormalization(center=True, scale=True))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Conv2D(16,kernel_size=(3, 3),activation='relu',padding='valid'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
        model.add(tf.keras.layers.BatchNormalization(center=True, scale=True))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        return model
    
    def normalize_image(self, img):
        frame = img / 255.
        return frame

    def get_samples(self):
        dir_ = self.dir_ + "samples/"
        samples = []
        for sample_name in os.listdir(dir_):
            frame = (cv2.imread(dir_ + sample_name))
            frame = cv2.resize(frame, (settings.eye_resolution, settings.eye_resolution))
            frame = self.normalize_image(frame)
            label = 1 if sample_name.split('_')[0][0].lower() == "o" else 0
            samples.append((frame, label, sample_name))
        return samples

    def train(self, train_set, val_set, test_set):
        (train_set, val_set, test_set) = helper.postprocess_dataset(train_set, val_set, test_set, self.default_val_ratio, self.default_test_ratio)
        helper.save_training_id(test_set, self.write_dir)

        model = self.get_model()

        saver = tf.keras.callbacks.ModelCheckpoint(filepath = self.write_dir + "model", save_weights_only = True, verbose = 1, monitor='val_loss', mode='min', save_best_only=True)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=self.patience, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

        cbs = [es, saver]
        if len(val_set) == 0:
            cbs = [tf.keras.callbacks.ModelCheckpoint(filepath = self.write_dir + "model", save_weights_only = True, verbose = 1)]

        # Fit model
        full_val_history = []
        full_train_history = []
        full_val_acc_history = []
        full_train_acc_history = []
        for i in range(self.iterations):
            history = model.fit(np.array([np.array(entry[0]) for entry in train_set]), np.array([entry[1] for entry in train_set]), batch_size=8, validation_data=(np.array([np.array(entry[0]) for entry in val_set]), np.array([entry[1] for entry in val_set])), epochs=(i+1) * self.interval, initial_epoch=i * self.interval, callbacks=cbs)
            if 'val_loss' in history.history:
                full_val_history = full_val_history + history.history['val_loss']
            full_train_history = full_train_history + history.history['loss']
            if 'val_accuracy' in history.history:
                full_val_acc_history = full_val_acc_history + history.history['val_accuracy']
            full_train_acc_history = full_train_acc_history + history.history['accuracy']
            model.load_weights(self.write_dir + "model")
        model.load_weights(self.write_dir + "model") # Load best model
        f = open(self.write_dir + "model.json", "w")
        f.write(model.to_json()) # Save architecture
        f.close()

        helper.save_learning_graphs(full_train_history, full_val_history, full_train_acc_history, full_val_acc_history, self.write_dir)

        train_results = helper.get_metrics(model, train_set, False)
        val_results = helper.get_metrics(model, val_set, False, self.write_dir + "val_results")
        test_results = helper.get_metrics(model, test_set, False, self.write_dir + "test_results")

        helper.get_conf_matrix(model, val_set, False, True, self.write_dir + "val_confusion")
        (test_labels, test_predictions) = helper.get_conf_matrix(model, test_set, False, True, self.write_dir + "test_confusion")

        return (train_results, val_results, test_results, test_labels, test_predictions)

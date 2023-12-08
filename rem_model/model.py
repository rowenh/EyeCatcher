import sys
import os
import tensorflow as tf
import cv2
import numpy as np
from keras.models import model_from_json

import pathlib
sys.path.append(pathlib.Path(__file__).parent.resolve())
import settings
import src.base_model
import src.model_helper as helper

class RemModel(src.base_model.BaseModel):
    learning_rate = 0.01
    iterations = 5 # Number of epoch changes
    interval = 5 # 50 # Number of epochs per learning rate
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
        # SPATIAL FEATURES
        model.add(tf.keras.layers.Conv3D(6, kernel_size=(1, 5, 5),strides=(1,2,2),activation='relu',input_shape=(6, 56, 56, 1),padding='same',data_format='channels_last'))
        #model.add(tf.keras.layers.Conv3D(16, kernel_size=(1, 3, 3),activation='relu',padding='same',data_format='channels_last'))
        model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2),padding='same',data_format='channels_last'))
        model.add(tf.keras.layers.BatchNormalization(center=True, scale=True))
        model.add(tf.keras.layers.Dropout(0.5))
        # TEMPORAL FEATURES
        model.add(tf.keras.layers.Conv3D(16, kernel_size=(3, 3, 3),strides=(2,1,1),activation='relu',padding='same',data_format='channels_last'))
        model.add(tf.keras.layers.Conv3D(16, kernel_size=(3, 3, 3),strides=(1,1,1),activation='relu',padding='valid',data_format='channels_last'))
        model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2),padding='same',data_format='channels_last'))
        model.add(tf.keras.layers.BatchNormalization(center=True, scale=True))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Flatten())
        # CLASSIFICATION
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        return model
    
    def normalize_image(self, img):
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame = cv2.equalizeHist(frame)
        frame = frame / 255.
        return frame

    def get_samples(self):
        dir_ = self.dir_ + "samples/"
        samples = []
        for sample_name in os.listdir(dir_):
            sample = []
            for frame_name in os.listdir(dir_ + sample_name):
                frame = (cv2.imread(dir_ + sample_name + "/" + frame_name))
                frame = cv2.resize(frame, (settings.eye_resolution, settings.eye_resolution))
                frame = self.normalize_image(frame)
                sample.append(frame)
            if len(sample) != settings.rem_frames:
                print("Unexpected input folder in REM samples! From " + sample_name + ".")
            label = 1 if sample_name.split('_')[0].endswith("r") else 0
            samples.append((sample, label, sample_name))
        return samples

    def lr_scheduler(self, epoch, lr):
        lr = self.learning_rate
        for _ in range(int(epoch / self.interval)):
            lr = lr / 2
        return lr

    def train(self, train_set, val_set, test_set):
        (train_set, val_set, test_set) = helper.postprocess_dataset(train_set, val_set, test_set, self.default_val_ratio, self.default_test_ratio)
        helper.save_training_id(test_set, self.write_dir)

        model = self.get_model()

        lrs = tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler)
        saver = tf.keras.callbacks.ModelCheckpoint(filepath = self.write_dir + "model", save_weights_only = True, verbose = 1, monitor='val_loss', mode='min', save_best_only=True)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=self.patience, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

        cbs = [lrs, es, saver]
        if len(val_set) == 0:
            cbs = [lrs, tf.keras.callbacks.ModelCheckpoint(filepath = self.write_dir + "model", save_weights_only = True, verbose = 1)]

        # Fit model
        full_val_history = []
        full_train_history = []
        full_val_acc_history = []
        full_train_acc_history = []
        for i in range(self.iterations):
            history = model.fit(np.array([np.array(entry[0]) for entry in train_set])[...,np.newaxis], np.array([entry[1] for entry in train_set]), batch_size=8, validation_data=(np.array([np.array(entry[0]) for entry in val_set])[...,np.newaxis], np.array([entry[1] for entry in val_set])), epochs=(i+1) * self.interval, initial_epoch=i * self.interval, callbacks=cbs)
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

        train_results = helper.get_metrics(model, train_set, True)
        val_results = helper.get_metrics(model, val_set, True, self.write_dir + "val_results")
        test_results = helper.get_metrics(model, test_set, True, self.write_dir + "test_results")

        helper.get_conf_matrix(model, val_set, True, True, self.write_dir + "val_confusion")
        (test_labels, test_predictions) = helper.get_conf_matrix(model, test_set, True, True, self.write_dir + "test_confusion")

        return (train_results, val_results, test_results, test_labels, test_predictions)

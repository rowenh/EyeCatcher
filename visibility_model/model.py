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
import open_model.model

class VisibilityModel(src.base_model.BaseModel):
    learning_rate = 0.001
    iterations = 1 # Number of epoch changes
    interval = 5 # 100 # Number of epochs per learning rate
    patience = 10 # Patience for early stopping
    default_test_ratio = 0.2 # Test ratio used if no test items are defined otherwise
    default_val_ratio = 0.15 # Val ratio used if no val items are defined otherwise

    def __init__(self):
        super().__init__(str(pathlib.Path(__file__).parent.resolve()).replace("\\", "/") + "/")
        self.oc = open_model.model.OpenModel()

    def load_model(self):
        f = open(self.read_dir + "model.json", "r")
        model = model_from_json(f.read())
        model.load_weights(self.read_dir + "model")
        return model

    def get_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(6,kernel_size=(5, 5),strides=2,activation='relu',input_shape=(56,56,1),padding='valid'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Conv2D(16,kernel_size=(3, 3),activation='relu',padding='valid'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
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

    def get_samples_oc(self):
        dir_ = self.oc.dir_ + "samples/"
        samples = []
        for sample_name in os.listdir(dir_):
            frame = (cv2.imread(dir_ + sample_name))
            frame = cv2.resize(frame, (settings.eye_resolution, settings.eye_resolution))
            frame = self.normalize_image(frame)
            label = 1 if sample_name.split('_')[0][0].lower() == "o" else 0
            samples.append((frame, label, sample_name))
        return samples

    def get_samples(self):
        path = self.dir_ + "samples/"
        samples = []
        for sample_name in os.listdir(path):
            frame = (cv2.imread(path + sample_name))
            frame = cv2.resize(frame, (settings.eye_resolution, settings.eye_resolution))
            frame = self.normalize_image(frame)
            label = 1 if sample_name.split('_')[0][0].lower() == "v" else 0
            samples.append((frame, label, sample_name))
        return samples

    def train(self, train_set, val_set, test_set):
        model = self.get_model()

        print("Initializing weights with open-closed task.")
        (train_set_oc, val_set_oc, test_set_oc) = helper.postprocess_dataset(self.get_samples_oc(), [], [], 0.15, 0)
        self.train_(model, train_set_oc, val_set_oc, test_set_oc)
        print("Initialization done.")

        (train_set, val_set, test_set) = helper.postprocess_dataset(train_set, val_set, test_set, self.default_val_ratio, self.default_test_ratio)
        helper.save_training_id(test_set, self.write_dir)
        self.train_(model, train_set, val_set, test_set)

    def train_(self, model, train_set, val_set, test_set):
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

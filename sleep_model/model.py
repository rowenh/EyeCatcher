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
import numpy as np
import tensorflow_decision_forests as tfdf
import matplotlib.pyplot as plt
from keras.models import model_from_json

import pathlib
sys.path.append(pathlib.Path(__file__).parent.resolve())
import settings as settings
import src.base_model
import src.model_helper as helper

class SleepModel(src.base_model.BaseModel):
    num_trees = 10
    frame_percentage_constraint = 0.5 # Percentage of frame predictions that are required to make a prediction on a window
    input_ids = {"c":0, "cr":1, "o":2, "or":3}
    output_ids = {"w":0, "as":1, "qs":2}

    def __init__(self):
        super().__init__(str(pathlib.Path(__file__).parent.resolve()).replace("\\", "/") + "/")

    def load_model(self):
        model = tf.keras.models.load_model(self.read_dir + "model")
        return model

    def get_model(self):
        model = tfdf.keras.RandomForestModel(num_trees = self.num_trees)
        model.compile(metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model

    def get_samples(self):
        dir_ = self.dir_ + "samples/"
        samples = []
        for annot_file in sorted(os.listdir(dir_),key=lambda x: (int(x.split('_')[-1]))):
            with open(dir_ + "/" + annot_file, 'r') as f:
                entries = list(map(str.strip, f.readlines()))
                if entries[0][0] == "?":
                    continue
                label = 0 if entries[1] not in self.output_ids else self.output_ids[entries[1]]
                input_vector = []
                input_dict = {}
                for l in entries[3:]:
                    input_dict[l.split(" ")[0]] = float(l.split(" ")[1])
                successful_percentage = 1 - (input_dict.get("x", 0) + (1 - input_dict.get("x", 0)) * input_dict.get("m", 0))
                if successful_percentage < self.frame_percentage_constraint:
                    continue
                for k in self.input_ids.keys():
                    input_vector.append(input_dict.get(k, 0))
                samples.append((input_vector, label, annot_file))
        return samples

    def train(self, train_set, val_set, test_set):
        helper.save_training_id(test_set, self.write_dir)

        model = self.get_model()

        model.fit(np.array([entry[0] for entry in train_set]), np.array([entry[1] for entry in train_set]))

        model.save(self.write_dir + "model")

        with open(self.write_dir + "tree.html", "w+") as html_file:
            html_file.write(tfdf.model_plotter.plot_model(model, tree_idx=0, max_depth=10))

        train_results = helper.get_metrics(model, [(entry[0], tf.keras.utils.to_categorical([entry[1]], num_classes=len(self.output_ids.keys()))[0], entry[2]) for entry in train_set], False)
        test_results = helper.get_metrics(model, [(entry[0], tf.keras.utils.to_categorical([entry[1]], num_classes=len(self.output_ids.keys()))[0], entry[2]) for entry in test_set], False, self.write_dir + "test_results")

        (test_labels, test_predictions) = helper.get_conf_matrix(model, test_set, False, False, self.write_dir + "test_confusion")
        np.savetxt(self.write_dir + "test_predictions", np.array(test_predictions))
        self.save_sleep_graph(test_set, test_predictions, self.write_dir + "test_sleepgraph.png")

        return (train_results, [], test_results, test_labels, test_predictions)
    
    def save_sleep_graph(self, test_set, test_predictions, output_path):
        if len(test_set) == 0:
            return
        X = []
        Y = []
        test_indices = [int(x[2].split("_")[-1]) for x in test_set]
        width = max(test_indices) + 1
        pred_index = 0
        for i in range(width):
            X.append(i)
            if i not in test_indices:
                Y.append(np.nan)
                continue
            Y.append(test_predictions[pred_index])
            X.append(i+1)
            Y.append(test_predictions[pred_index])
            pred_index = pred_index + 1
        plt.clf()
        plt.plot(X, Y, color='blue')
        plt.axis([0, width, -1, 3])
        plt.yticks([-1, 0, 1, 2, 3], ['', 'W', 'AS', 'QS', ''])
        plt.xlabel('Timestamp (in minutes)')
        plt.savefig(output_path)

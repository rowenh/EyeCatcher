import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def get_metrics(model, set_, expand_dims, output_path=None):
    results = []
    if len(set_) > 0:
        inp = np.array([np.array(entry[0]) for entry in set_])
        results = model.evaluate(inp if not expand_dims else inp[...,np.newaxis], np.array([entry[1] for entry in set_]), verbose=2)
        precision = results[3]
        recall = results[4]
        results.append(2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))) # F1
        if output_path is not None:
            with open(output_path, "w") as f:
                f.write("(Loss, Acc, AUC, Precision, Recall, F1): " + str(results))
    return results

def get_conf_matrix(model, set_, expand_dims, binary, output_path=None):
    labels = []
    predictions = []
    if len(set_) > 0:
        inp = np.array([np.array(entry[0]) for entry in set_])
        preds_raw = model.predict(inp if not expand_dims else inp[...,np.newaxis])
        if binary:
            predictions = (preds_raw > 0.5).astype("int32")
            predictions = [p[0] for p in predictions]
        else:
            predictions = tf.argmax(preds_raw, 1).numpy().tolist()
        labels = [entry[1] for entry in set_]
        matrix = tf.math.confusion_matrix(np.array(labels), np.array(predictions))
        if output_path is not None:
            np.savetxt(output_path, matrix)
    return (labels, predictions)

def save_learning_graphs(full_train_history, full_val_history, full_train_acc_history, full_val_acc_history, output_directory):
    epochs = range(1, len(full_train_history) + 1)

    if len(full_val_history) == 0:
        full_val_history = [float('inf') for _ in full_train_history]
    if len(full_val_acc_history) == 0:
        full_val_acc_history = [float('inf') for _ in full_train_acc_history]
    
    plt.clf()
    plt.plot(epochs, full_train_history, 'C1', label='Training loss')
    plt.plot(epochs, full_val_history, 'C2', label='Validation loss')
    plt.ylim(0, 2)
    plt.legend()
    plt.savefig(output_directory + "loss_graph.png")

    plt.clf()
    plt.plot(epochs, full_train_acc_history, 'C1', label='Training accuracy')
    plt.plot(epochs, full_val_acc_history, 'C2', label='Validation accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(output_directory + "acc_graph.png")

def save_training_id(test_set, output_directory):
    f = open(output_directory + "training_id", "w")
    f.write("\n".join([s[2] for s in test_set]))
    f.close()

def postprocess_dataset(train_set, val_set, test_set, val_ratio, test_ratio):
    # Apply default val and test ratios, when applicable
    classes = list(set([x[1] for x in train_set]))
    samples_per_class = [None for _ in classes]
    for i in range(len(classes)):
        samples_per_class[i] = [x for x in train_set if x[1] == classes[i]]

    if len(val_set) == 0 and val_ratio > 0:
        val_indices = []
        for lst in samples_per_class:
            val_indices = val_indices + random.sample(range(len(train_set)), int(len(lst) * val_ratio))
        val_set = [train_set[i] for i in range(len(train_set)) if i in val_indices]
        train_set = [train_set[i] for i in range(len(train_set)) if i not in val_indices]

    if len(test_set) == 0 and test_ratio > 0:
        test_indices = []
        for lst in samples_per_class:
            test_indices = test_indices + random.sample(range(len(train_set)), int(len(lst) * test_ratio))
        test_set = [train_set[i] for i in range(len(train_set)) if i in test_indices]
        train_set = [train_set[i] for i in range(len(train_set)) if i not in test_indices]

    return (train_set, val_set, test_set)

def kfold(model, samples, folds, fold_names):
    (kfold_train_results, kfold_val_results, kfold_test_results, kfold_test_labels, kfold_test_predictions) = ([], [], [], [], [])
    fold_index = 0
    kfold_alltestresults = []
    for fold in folds:
        test_samples = [samples[i] for i in fold]
        train_samples = [samples[i] for i in range(len(samples)) if i not in fold]
        (train_results, val_results, test_results, test_labels, test_predictions) = model.train(train_samples, [], test_samples)
        kfold_train_results.append(train_results)
        kfold_val_results.append(val_results)
        kfold_test_results.append(test_results)
        kfold_alltestresults.append(test_results)
        kfold_test_labels = kfold_test_labels + test_labels
        kfold_test_predictions = kfold_test_predictions + test_predictions
        fold_index = fold_index + 1

    (kfold_train_results, kfold_val_results, kfold_test_results) = (np.swapaxes(kfold_train_results, 0, 1), np.swapaxes(kfold_val_results, 0, 1) if len(kfold_val_results[0]) > 0 else [], np.swapaxes(kfold_test_results, 0, 1))

    kfold_testlabels = kfold_test_labels
    kfold_testpredictions = kfold_test_predictions
    kfold_testaverages = [sum(test_results) / len(test_results) for test_results in kfold_test_results]
    kfold_valaverages = [sum(val_results) / len(val_results) for val_results in kfold_val_results]
    kfold_trainaverages = [sum(train_results) / len(train_results) for train_results in kfold_train_results]
    kfold_testsd = [np.std(test_results) for test_results in kfold_test_results]
    kfold_valsd = [np.std(val_results) for val_results in kfold_val_results]
    kfold_trainsd = [np.std(train_results) for train_results in kfold_train_results]

    kfold_output_path = model.write_dir + "/kfold/"
    if not os.path.exists(kfold_output_path):
        os.mkdir(kfold_output_path)
    with open(kfold_output_path + "metrics", "w") as f:
        f.write("Test Averages: " + str(kfold_testaverages))
        f.write("\nTest SD: " + str(kfold_testsd))
        f.write("\nVal Averages: " + str(kfold_valaverages))
        f.write("\nVal SD: " + str(kfold_valsd))
        f.write("\nTrain Averages: " + str(kfold_trainaverages))
        f.write("\nTrain SD: " + str(kfold_trainsd))

        f.write("\n")
        i = 0
        for _ in kfold_alltestresults:
            f.write("\nTest Results (fold " + fold_names[i] + "): " + str(kfold_alltestresults[i]))
            i = i + 1
        np.savetxt(kfold_output_path + "confusion", tf.math.confusion_matrix(np.array(kfold_testlabels), np.array(kfold_testpredictions)))

def run(model, config):
    train_set = config["train_set"]
    val_set = config["val_set"]
    test_set = config["test_set"]
    folds = config["folds"]

    samples = model.get_samples() # Gives array of (Input, Label, Name)
    shuffled_indices = list(range(len(samples)))
    random.shuffle(shuffled_indices)
    samples = [samples[i] for i in shuffled_indices]

    if folds is not None: # Cross validation
        indices_per_fold = []
        if folds == "loocv":
            indices_per_fold = [[item] for item in range(len(samples))]
            folds = [x[2] for x in samples]
        else:
            folds = folds.split("=")
            indices_per_fold = [[] for _ in folds]
            for i in range(len(samples)):
                (add, fold_number) = (False, 0)
                for j in range(len(folds)):
                    fold = folds[j].split("+")
                    for k in range(len(fold)):
                        if fold[k] in samples[i][2]:
                            (add, fold_number) = (True, j)
                if add:
                    indices_per_fold[fold_number].append(i)
        kfold(model, samples, indices_per_fold, folds)
    else:
        (train_constraints, val_constraints, test_constraints) = ([] if train_set is None else train_set.split("+"), [] if val_set is None else val_set.split("+"), [] if test_set is None else test_set.split("+"))
        (train_samples, val_samples, test_samples) = ([samples[i] for i in range(len(samples)) if any(c in samples[i][2] for c in train_constraints)] if len(train_constraints) > 0 else 
            [samples[i] for i in range(len(samples)) if not any(c in samples[i][2] for c in val_constraints) and not any(c in samples[i][2] for c in test_constraints)]
            , [samples[i] for i in range(len(samples)) if any(c in samples[i][2] for c in val_constraints)]
            , [samples[i] for i in range(len(samples)) if any(c in samples[i][2] for c in test_constraints)])
        model.train(train_samples, val_samples, test_samples)

import numpy as np
import csv
import random


def preprocess(metrics, recalculate=False, causal=False):

    categories, data = clean_data()
    if recalculate:
        training_data, training_labels, test_data, test_labels = split_data(data, categories, 0.2, causal=causal)
        print("Recalculating data...")
    else:
        try:
            training_data = np.load("COMPAS_train_data.npy")
            training_labels = np.load("COMPAS_train_labels.npy")
            test_data = np.load("COMPAS_test_data.npy")
            test_labels = np.load("COMPAS_test_labels.npy")
            for i in range(len(training_labels)):
                training_labels[i] = int(training_labels[i])
            for i in range(len(test_labels)):
                test_labels[i] = int(test_labels[i])
            data = np.concatenate((training_data, test_data))
            print("Loaded training data")

        except:
            training_data, training_labels, test_data, test_labels = split_data(data, categories, 0.2, causal=causal)
            print("Could not locate data...")

    used_metrics = metrics
    training_data, reduced_categories, training_predictions = reduce_data(categories, training_data, used_metrics)
    np.save("COMPAS_train_decile_scores", training_predictions)
    test_data, reduced_categories, test_predictions = reduce_data(categories, test_data, used_metrics)
    np.save("COMPAS_test_decile_scores", test_predictions)
    mappings = determine_mappings(data, used_metrics)
    vectorize_data(training_data, reduced_categories, metrics, mappings)
    vectorize_data(test_data, reduced_categories, metrics, mappings)
    vectorize_labels(training_labels)
    vectorize_labels(test_labels)

    training_data = np.array(training_data)
    test_data = np.array(test_data)
    training_labels = np.array(training_labels)
    test_labels = np.array(test_labels)

    return training_data, training_labels, test_data, test_labels, reduced_categories, mappings


def metric_vs_recid(metric):
    with open("compas-scores-two-years.csv", "r+") as compas_data:
        reader = csv.reader(compas_data)
        totals = {}
        possible_values = {}
        is_recid = 52
        index = -1
        categories = reader.__next__()
        for i in range(len(categories)):
            if metric in categories[i]:
                index = i

        if index == -1:
            print("Couldn't find metric: " + metric)
            return

        row = reader.__next__()
        while row is not None:

            if row[is_recid] != "-1":
                if row[index] in possible_values:
                    possible_values[row[index]] = int(possible_values[row[index]]) + int(row[is_recid])
                    totals[row[index]] = int(totals[row[index]]) + 1
                else:
                    possible_values[row[index]] = row[is_recid]
                    totals[row[index]] = 1

            try:
                row = reader.__next__()
            except:
                break

        for value in possible_values:
            print(str(value) + ": " + str(int(possible_values[value])*100/int(totals[value])))
        print("")


def clean_data():
    pos_data = []
    neg_data = []
    # Reads data from csv into a list of lists
    # Throws out any rows with a -1 for recidivism
    with open("compas-scores-two-years.csv", "r+") as compas_data:
        is_recid = 52
        reader = csv.reader(compas_data)
        categories = reader.__next__()
        row = reader.__next__()
        while True:

            if row[is_recid] != "-1":
                if row[is_recid] == "0":
                    neg_data.append(row)
                else:
                    pos_data.append(row)

            try:
                row = reader.__next__()
            except:
                break

        if len(pos_data) < len(neg_data):
            data = pos_data + random.sample(neg_data, len(pos_data))
        else:
            data = neg_data + random.sample(pos_data, len(neg_data))

    random.shuffle(data)

    return categories, data


def split_data(data, categories, percent_test, causal=False):

    if causal:
        data = enforce_causal_discrimination(data, categories, "race", "Caucasian")

    is_recid = 52

    sample_size = int(percent_test * len(data))

    while True:
        training_data = data[:-sample_size]
        test_data = data[-sample_size:]

        training_labels = []
        test_labels = []

        for i in range(len(training_data)):
            training_labels.append(training_data[i][is_recid])

        zeros = 0
        ones = 0
        for i in range(len(test_data)):
            if test_data[i][is_recid] == "0":
                zeros += 1
            else:
                ones += 1
            test_labels.append(test_data[i][is_recid])

        if zeros == ones:
            break
        else:
            random.shuffle(data)

    np.save("COMPAS_train_data", training_data)
    np.save("COMPAS_train_labels", training_labels)
    np.save("COMPAS_test_data", test_data)
    np.save("COMPAS_test_labels", test_labels)
    return training_data, training_labels, test_data, test_labels


def vectorize_data(data, categories, metrics, mappings):

    for metric in metrics:
        index = -1
        for i in range(len(categories)):
            if metric in categories[i]:
                index = i
                break

        for i in range(len(data)):
            data[i][index] = mappings[metric][data[i][index]]


def vectorize_labels(labels):
    for i in range(len(labels)):
        labels[i] = int(labels[i])


def reduce_data(categories, data, keep_metrics):
    metric_indices = []
    reduced_categories = []
    for metric in keep_metrics:
        metric_indices.append(categories.index(metric))

    prediction_index = -1
    for i in range(len(categories)):
        if "decile_score" in categories[i]:
            prediction_index = i
    predictions = []

    reduced_data = []
    for i in range(len(data)):
        row = []
        for index in metric_indices:
            row.append(data[i][index])
        reduced_data.append(row)
        predictions.append(data[i][prediction_index])

    for index in metric_indices:
        reduced_categories.append(categories[index])

    return reduced_data, reduced_categories, predictions


def determine_mappings(data, keep_metrics):

    with open("compas-scores-two-years.csv", "r+") as compas_data:
        #print("Opened data file")
        mappings = {}
        reader = csv.reader(compas_data)
        index = -1
        categories = reader.__next__()
        for metric in keep_metrics:
            mappings[metric] = {}
            for i in range(len(categories)):
                if metric in categories[i]:
                    index = i
                    break

            if index == -1:
                print("Couldn't find metric: " + metric)
                return

            possible_values = set()
            for i in range(len(data)):
                possible_values.add(data[i][index])

            for i, value in enumerate(sorted(possible_values)):
                mappings[metric][value] = i

    return mappings


def enforce_causal_discrimination(data, categories, reference_metric, reference_value):
    index = categories.index(reference_metric)
    augmented_data = list.copy(data)

    # Loop through training data and add an entry for each class besides the reference class
    for i, row in enumerate(data):
        if row[index] != reference_value:
            duplicate = list.copy(row)
            duplicate[index] = reference_value
            augmented_data.append(duplicate)

    return augmented_data

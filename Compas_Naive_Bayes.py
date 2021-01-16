from sklearn.naive_bayes import MultinomialNB
import numpy as np
from Preprocessing import preprocess
from Report_Results import report_results
from utils import *
import time
start = time.time()


def naive_bayes_classification(metrics):
    """This function takes as input a list of metrics and performs the Naive Bayes classification.
    :param metrics: List containing the metrics that we want to use for classification."""

    training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

    NBC = MultinomialNB()
    NBC.fit(training_data, training_labels)

    data = np.concatenate((training_data, test_data))
    labels = np.concatenate((training_labels, test_labels))

    class_predictions = NBC.predict_proba(data)
    predictions = []

    for i in range(len(labels)):
        predictions.append(class_predictions[i][1])

    return data, predictions, labels, categories, mappings


metrics = ["race", "sex", "age", 'c_charge_degree', 'priors_count', 'c_charge_desc']
data, predictions, labels, categories, mappings = naive_bayes_classification(metrics)
race_cases = get_cases_by_metric(data, categories, "race", mappings, predictions, labels)

report_results(race_cases)
end = time.time()
print('Runtime:', start - end)

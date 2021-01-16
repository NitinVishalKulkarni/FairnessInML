from sklearn import svm
from Preprocessing import preprocess
from Report_Results import report_results
import numpy as np
from utils import *


def SVM_classification(metrics_):
    """This function takes as input  a list of metrics and uses a Support vector machine to perform the classification.
      :param metrics_: List containing the metrics that we want to use for classification."""

    training_data, training_labels, test_data, test_labels, categories, mappings = \
        preprocess(metrics_, recalculate=False, causal=False)

    np.random.seed(42)
    SVR = svm.LinearSVR(C=1.0/float(len(test_data)), max_iter=5000)
    SVR.fit(training_data, training_labels)

    data = np.concatenate((training_data, test_data))
    labels = np.concatenate((training_labels, test_labels))

    predictions = SVR.predict(data)

    return data, predictions, labels, categories, mappings


metrics = ["sex", "age_cat", 'race', 'c_charge_degree', 'priors_count']

data, predictions, labels, categories, mappings = SVM_classification(metrics)
race_cases = get_cases_by_metric(data, categories, "race", mappings, predictions, labels)

report_results(race_cases)


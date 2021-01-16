import matplotlib.pyplot as plt


def get_cases_by_metric(test_data, categories, metric, mappings, predictions, labels):
    """ Groups all cases by metric, creating a dictionary with each group value as a key
    @:param test_data:      List of lists, the full data set to pull groups from
    @:param categories:     List of column titles, stored separately from the numerical data
    @:param metric:         String used to determine groups, i.e 'race'
    @:param mappings:       Dictionary mappings from training, used to convert the data between numerical and string format
    @:param predictions:    List of predicted values produced from a machine learning model.
    @:param labels:         List of labels for the test data

    @:note: test_data, predictions, and labels should correspond to one another
    @:note: Groups with less than 50 members are not considered sufficiently representative and are excluded.

    @:returns total_cases:  Dictionary with each group value as keys. Each key has a list of (prediction, label) tuples
    representing all of the data points within that group."""

    total_cases = {}
    index = -1
    for i in range(len(categories)):
        if metric in categories[i]:
            index = i
            break

    for value in mappings[metric].keys():
        cases = []
        for i in range(len(test_data)):
            if test_data[i][index] == mappings[metric][value]:
                cases.append((float(predictions[i]), int(labels[i])))

        # Only include groups that have more than 50 members
        if len(cases) > 50:
            total_cases[value] = cases

    return total_cases


def apply_threshold(prediction_label_pairs, threshold):
    """Applies a threshold to real-valued model predictions to make them either 0 or 1. Values above the threshold become
    1's, values below or equal to the threshold become 0's.

    @:param predictions:    Tuples of the form (prediction, label), such as those returned by get_cases_by_metric
    @:param threshold:      Float or Int value used to calculate the predicted value

    @:returns predictions:  The thresholded version of the same input (prediction, label) tuples."""

    threshed = [(0, 0)] * len(prediction_label_pairs)
    for i in range(len(prediction_label_pairs)):
        if prediction_label_pairs[i][0] <= threshold:
            threshed[i] = (0, prediction_label_pairs[i][1])
        else:
            threshed[i] = (1, prediction_label_pairs[i][1])

    return threshed


def get_total_accuracy(classifications):
    """Gets the total accuracy of a set of classifications

    @:param classifications:    a dictionary of all the classifications, separated into groups. Each group contains
                                a list of (prediction, label) tuples

    @:note:                     assumes that the predictions have been already thresholded

    @:returns total_accuracy:   the total accuracy of the classifications."""

    total_correct = 0.0
    total_num_cases = 0.0
    for group in classifications.keys():
        for prediction, label in classifications[group]:
            total_num_cases += 1.0
            if prediction == label:
                total_correct += 1.0

    return total_correct / total_num_cases


def get_num_correct(prediction_label_pairs):
    """Determines the number of correct predictions in a group

    @:param prediction_label_pairs:       List of (prediction, label) tuples

    @:note:             Assumes predictions have already been thresholded

    @:returns num_correct:  Int value of correct predictions. Dividing this by len(category) would give the
                            accuracy for the group."""

    num_correct = 0
    for pair in prediction_label_pairs:
        prediction = int(pair[0])
        label = int(pair[1])
        if prediction == label:
            num_correct += 1

    return num_correct


def get_num_false_positives(prediction_label_pairs):
    """Determines the number of false positives in a group

    @:param prediction_label_pairs:   List of (prediction, label) tuples

    @:note:             Assumes predictions have already been thresholded

    @:returns false_positives:        The number of false positives (prediction == 1, label == 0)."""

    false_positives = 0

    for pair in prediction_label_pairs:
        prediction = int(pair[0])
        label = int(pair[1])
        if prediction == 1 and label == 0:
            false_positives += 1

    return false_positives


def get_false_positive_rate(prediction_label_pairs):
    """Determines the rate of false positives in a group

    @:param prediction_label_pairs:     List of (prediction, label) tuples

    @:note:                             Assumes predictions have already been thresholded

    @:returns FPR:                      The number of false positives divided by the number of labelled negatives. Will
                                        return 0 to avoid divide by 0, but in practice there should be no instances of no
                                        labelled negatives."""

    false_positives = 0
    labelled_negatives = 0

    for pair in prediction_label_pairs:
        prediction = int(pair[0])
        label = int(pair[1])
        if label == 0:
            labelled_negatives += 1
            if prediction == 1:
                false_positives += 1

    if labelled_negatives != 0:
        return false_positives / labelled_negatives
    else:
        return 0


def get_num_true_negatives(prediction_label_pairs):
    """Determines the number of true negatives in a group

    @:param prediction_label_pairs:     List of (prediction, label) tuples

    @:note:                             Assumes predictions have already been thresholded

    @:returns true_negatives            The number of true negatives (prediction == 0, label == 0)."""

    true_negatives = 0

    for pair in prediction_label_pairs:
        prediction = int(pair[0])
        label = int(pair[1])
        if prediction == 0 and label == 0:
            true_negatives += 1

    return true_negatives


def get_true_negative_rate(prediction_labels_pairs):
    """Determines the rate of true negatives in a group

    @:param prediction_label_pairs:     List of (prediction, label) tuples

    @:note:             Assumes predictions have already been thresholded

    @:returns TNR:                      1 - false_positive_rate."""
    return 1 - get_false_positive_rate(prediction_labels_pairs)


def get_num_false_negatives(prediction_label_pairs):
    """Determines the number of false negatives in a group

    @:param prediction_label_pairs:     List of (prediction, label) tuples

    @:note:                             Assumes predictions have already been thresholded

    @:returns false_negatives           The number of false negatives (prediction == 0, label == 1)."""

    false_negatives = 0

    for pair in prediction_label_pairs:
        prediction = int(pair[0])
        label = int(pair[1])
        if prediction == 0 and label == 1:
            false_negatives += 1

    return false_negatives


def get_false_negative_rate(prediction_label_pairs):
    """Determines the rate of false negatives in a group

    @:param prediction_label_pairs:     List of (prediction, label) tuples

    @:note:                             Assumes predictions have already been thresholded

    @:returns FNR:                      The number of false negatives divided by the number of labelled positives. Will
                                        return 0 to avoid divide by 0, but in practice there should be no instances of no
                                        labelled positives."""

    false_negatives = 0
    labelled_positives = 0

    for pair in prediction_label_pairs:
        prediction = int(pair[0])
        label = int(pair[1])
        if label == 1:
            labelled_positives += 1
            if prediction == 0:
                false_negatives += 1

    if labelled_positives != 0:
        return false_negatives / labelled_positives
    else:
        return 0


def get_num_true_positives(prediction_label_pairs):
    """Determines the number of true positives in a group

    @:param prediction_label_pairs:     List of (prediction, label) tuples

    @:note:                             Assumes predictions have already been thresholded

    @:returns true_positives           The number of true positives (prediction == 1, label == 1)."""
    true_positives = 0

    for pair in prediction_label_pairs:
        prediction = int(pair[0])
        label = int(pair[1])
        if prediction == 1 and label == 1:
            true_positives += 1

    return true_positives


def get_true_positive_rate(category):
    """Determines the rate of true positives in a group

    @:param prediction_label_pairs:     List of (prediction, label) tuples

    @:note:                             Assumes predictions have already been thresholded

    @:returns TPR:                      1 - false_negative_rate."""

    return 1 - get_false_negative_rate(category)



def get_num_predicted_positives(prediction_label_pairs):

    """Determines the number of samples that have a positive prediction

    @:param prediction_label_pairs:     List of (prediction, label) tuples

    @:note:                             Assumes predictions have already been thresholded

    @:returns predicted_positives       Number of samples with a positive prediction."""

    predicted_positives = 0

    for pair in prediction_label_pairs:
        prediction = int(pair[0])
        if prediction == 1:
            predicted_positives += 1

    return predicted_positives


def get_positive_predictive_value(prediction_label_pairs):
    """Determines the positive predictive value of a group, defined as the number of true positives divided by the
    number of predicted positives

    @:param prediction_label_pairs:     List of (prediction, label) tuples

    @:note:                             Assumes predictions have already been thresholded

    @:returns PPV:                      true positives / predicted positives."""

    true_positives = get_num_true_positives(prediction_label_pairs)
    predicted_positives = get_num_predicted_positives(prediction_label_pairs)

    if predicted_positives == 0:
        return 0
    else:
        return true_positives / predicted_positives


def calculate_Fscore(prediction_label_pairs):
    """Calculates the Fscore (or harmonic mean) of a group. Used as a substitute for accuracy when data is skewed

    @:param prediction_label_pairs:     List of (prediction, label) tuples

    @:note:                             Assumes predictions have already been thresholded

    @:returns Fscore                    Harmonic mean, defined as 2 * (precision * recall) + (precision + recall)."""

    precision = get_positive_predictive_value(prediction_label_pairs)
    recall = get_true_positive_rate(prediction_label_pairs)

    numerator = precision * recall
    denominator = precision + recall

    return 2 * (numerator/denominator)


def get_ROC_data(prediction_label_pairs, group):
    true_positives = []
    false_positives = []
    for i in range(1, 101):
        threshold = float(i) / 100.0
        eval_copy = list.copy(prediction_label_pairs)
        eval_copy = apply_threshold(eval_copy, threshold)
        TPR = get_true_positive_rate(eval_copy)
        FPR = get_false_positive_rate(eval_copy)
        true_positives.append(TPR)
        false_positives.append(FPR)

    return true_positives, false_positives, group


def plot_ROC_data(ROC_data_list):
    for curve in  ROC_data_list:
        TPR = curve[0]
        FPR = curve[1]
        title = curve[2]
        plt.plot(FPR, TPR, label=title)

    plt.legend()
    axes = plt.gca()
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.0])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    plt.show()


def apply_financials(data, group_level=False):

    # Costs for the various categories
    tp_val = -60076
    tn_val = 23088
    fp_val = -110076
    fn_val = -202330

    full_list = []
    if group_level:
        full_list = data
    else:
        for group in data.keys():
            full_list += data[group]

    num_tp = get_num_true_positives(full_list)
    num_tn = get_num_true_negatives(full_list)
    num_fp = get_num_false_positives(full_list)
    num_fn = get_num_false_negatives(full_list)

    total = 0.0
    total += num_tp * tp_val
    total += num_tn * tn_val
    total += num_fp * fp_val
    total += num_fn * fn_val

    return total



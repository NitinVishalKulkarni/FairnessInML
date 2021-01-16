from utils import *


def compare_probability(prob1, prob2, epsilon):
    return abs(prob1 - prob2) <= epsilon


def enforce_demographic_parity(categorical_results, epsilon):
    """ Determines the thresholds such that each group has equal predictive positive rates within
        a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
        a nontrivial solution with epsilon=0.02.
        Chooses the best solution of those that satisfy this constraint based on chosen
        secondary optimization criteria."""

    demographic_parity_data = {}
    thresholds = {}

    african_american_v = []
    caucasian_v = []
    hispanic_v = []
    other_v = []
    # Getting the prediction label pairs for all groups.
    for i in range(1, 101):
        threshold = float(i) / 100.0
        african_american = apply_threshold(categorical_results['African-American'], threshold)
        caucasian = apply_threshold(categorical_results['Caucasian'], threshold)
        hispanic = apply_threshold(categorical_results['Hispanic'], threshold)
        other = apply_threshold(categorical_results['Other'], threshold)
        african_american_v.append(african_american)
        caucasian_v.append(caucasian)
        hispanic_v.append(hispanic)
        other_v.append(other)

    # Getting the Probability of Positive Prediction for all groups.
    positive_pred_prob_african_american = []
    positive_pred_prob_caucasian = []
    positive_pred_prob_hispanic = []
    positive_pred_prob_other = []
    for i in range(len(african_american_v)):
        num_positive_predictions = get_num_predicted_positives(african_american_v[i])
        pos_pred_value = (num_positive_predictions / len(african_american_v[i]))
        if pos_pred_value > 0.5:
            positive_pred_prob_african_american.append(pos_pred_value)
    for i in range(len(caucasian_v)):
        num_positive_predictions = get_num_predicted_positives(caucasian_v[i])
        pos_pred_value = (num_positive_predictions / len(caucasian_v[i]))
        if pos_pred_value > 0.5:
            positive_pred_prob_caucasian.append(pos_pred_value)
    for i in range(len(hispanic_v)):
        num_positive_predictions = get_num_predicted_positives(hispanic_v[i])
        pos_pred_value = (num_positive_predictions / len(hispanic_v[i]))
        if pos_pred_value > 0.5:
            positive_pred_prob_hispanic.append(pos_pred_value)
    for i in range(len(other_v)):
        num_positive_predictions = get_num_predicted_positives(other_v[i])
        pos_pred_value = num_positive_predictions / len(other_v[i])
        if pos_pred_value > 0.5:
            positive_pred_prob_other.append(pos_pred_value)

    # Getting possible threshold values.
    possible_thresholds = []
    for i in range(len(positive_pred_prob_african_american)):
        for j in range(len(positive_pred_prob_caucasian)):
            a = compare_probability(positive_pred_prob_african_american[i], positive_pred_prob_caucasian[j], epsilon)
            if a is True:
                for k in range(len(positive_pred_prob_hispanic)):
                    a = compare_probability(positive_pred_prob_african_american[i], positive_pred_prob_hispanic[k],
                                            epsilon)
                    b = compare_probability(positive_pred_prob_caucasian[j], positive_pred_prob_hispanic[k], epsilon)
                    if a and b is True:
                        for m in range(len(positive_pred_prob_other)):
                            a = compare_probability(positive_pred_prob_african_american[i], positive_pred_prob_other[m],
                                                    epsilon)
                            b = compare_probability(positive_pred_prob_caucasian[j], positive_pred_prob_other[m],
                                                    epsilon)
                            c = compare_probability(positive_pred_prob_hispanic[k], positive_pred_prob_other[m],
                                                    epsilon)
                            if a and b and c is True:
                                possible_thresholds.append([(i + 1)/100, (j + 1)/100, (k + 1)/100, (m + 1)/100])

    # Optimizing for accuracy as the secondary metric.
    # Getting the accuracies for all the possible threshold values.
    accuracy_values = []
    for i in range(len(possible_thresholds)):
        african_american = apply_threshold(categorical_results['African-American'], (possible_thresholds[i][0]))
        caucasian = apply_threshold(categorical_results['Caucasian'], (possible_thresholds[i][1]))
        hispanic = apply_threshold(categorical_results['Hispanic'], (possible_thresholds[i][2]))
        other = apply_threshold(categorical_results['Other'], (possible_thresholds[i][3]))
        demographic_parity_data['African-American'] = african_american
        demographic_parity_data['Caucasian'] = caucasian
        demographic_parity_data['Hispanic'] = hispanic
        demographic_parity_data['Other'] = other
        accuracy_values.append(get_total_accuracy(demographic_parity_data))

    # Finding the maximum accuracy.
    maximum = max(accuracy_values)

    # Getting the final threshold values for the groups.
    threshold_values = []
    for value in range(len(accuracy_values)):
        if accuracy_values[value] == maximum:
            threshold_values = value

    # Storing the thresholds in a dictionary.
    thresholds['African-American'] = (possible_thresholds[threshold_values][0])
    thresholds['Caucasian'] = (possible_thresholds[threshold_values][1])
    thresholds['Hispanic'] = (possible_thresholds[threshold_values][2])
    thresholds['Other'] = (possible_thresholds[threshold_values][3])

    # Applying the thresholds on different groups.
    african_american = apply_threshold(categorical_results['African-American'], thresholds['African-American'])
    caucasian = apply_threshold(categorical_results['Caucasian'], thresholds['Caucasian'])
    hispanic = apply_threshold(categorical_results['Hispanic'], thresholds['Hispanic'])
    other = apply_threshold(categorical_results['Other'], thresholds['Other'])

    # Storing the data.
    demographic_parity_data['African-American'] = african_american
    demographic_parity_data['Caucasian'] = caucasian
    demographic_parity_data['Hispanic'] = hispanic
    demographic_parity_data['Other'] = other

    # Must complete this function!
    return demographic_parity_data, thresholds


def enforce_equal_opportunity(categorical_results, epsilon):
    """ Determine thresholds such that all groups have equal TPR within some tolerance value epsilon,
        and chooses best solution according to chosen secondary optimization criteria. For the Naive
        Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01."""

    thresholds = {}
    equal_opportunity_data = {}

    # Must complete this function!
    # Getting the true positive rates for all groups.
    true_positive = {}
    aa = []
    cc = []
    hi = []
    ot = []
    for i in categorical_results:
        roc_data = (get_ROC_data(prediction_label_pairs=categorical_results[i], group=i))
        if i == 'African-American':
            for x in roc_data[0]:
                if x > 0.9:
                    aa.append(x)
        if i == 'Caucasian':
            for x in roc_data[0]:
                if x > 0.9:
                    cc.append(x)
        if i == 'Hispanic':
            for x in roc_data[0]:
                if x > 0.9:
                    hi.append(x)
        if i == 'Other':
            for x in roc_data[0]:
                if x > 0.9:
                    ot.append(x)
    true_positive['African-American'] = aa
    true_positive['Caucasian'] = cc
    true_positive['Hispanic'] = hi
    true_positive['Other'] = ot

    print(true_positive)
    # Getting possible threshold values.
    possible_thresholds = []
    for i in range(len(true_positive['African-American'])):
        for j in range(len(true_positive['Caucasian'])):
            a = compare_probability(true_positive['African-American'][i], true_positive['Caucasian'][j], epsilon)
            if a is True:
                for k in range(len(true_positive['Hispanic'])):
                    a = compare_probability(true_positive['African-American'][i], true_positive['Hispanic'][k], epsilon)
                    b = compare_probability(true_positive['Caucasian'][j], true_positive['Hispanic'][k], epsilon)
                    if a and b is True:
                        for m in range(len(true_positive['Other'])):
                            a = compare_probability(true_positive['African-American'][i], true_positive['Other'][m], epsilon)
                            b = compare_probability(true_positive['Caucasian'][j], true_positive['Other'][m], epsilon)
                            c = compare_probability(true_positive['Hispanic'][k], true_positive['Other'][m], epsilon)
                            if a and b and c is True:
                                possible_thresholds.append([(i + 1)/100, (j + 1)/100, (k + 1)/100, (m + 1)/100])

    print('Possible Thresholds', len(possible_thresholds))
    # Optimizing for accuracy as the secondary metric.
    # Getting the accuracies for all the possible threshold values.
    accuracy_values = []
    for i in range(len(possible_thresholds)):
        african_american = apply_threshold(categorical_results['African-American'], (possible_thresholds[i][0]))
        caucasian = apply_threshold(categorical_results['Caucasian'], (possible_thresholds[i][1]))
        hispanic = apply_threshold(categorical_results['Hispanic'], (possible_thresholds[i][2]))
        other = apply_threshold(categorical_results['Other'], (possible_thresholds[i][3]))
        equal_opportunity_data['African-American'] = african_american
        equal_opportunity_data['Caucasian'] = caucasian
        equal_opportunity_data['Hispanic'] = hispanic
        equal_opportunity_data['Other'] = other
        accuracy_values.append(get_total_accuracy(equal_opportunity_data))

    # Finding the maximum accuracy.
    maximum = max(accuracy_values)

    # Getting the final threshold values for the groups.
    threshold_values = []
    for value in range(len(accuracy_values)):
        if accuracy_values[value] == maximum:
            threshold_values = value

    # Storing the thresholds in a dictionary.
    thresholds['African-American'] = (possible_thresholds[threshold_values][0])
    thresholds['Caucasian'] = (possible_thresholds[threshold_values][1])
    thresholds['Hispanic'] = (possible_thresholds[threshold_values][2])
    thresholds['Other'] = (possible_thresholds[threshold_values][3])

    # Applying the thresholds on different groups.
    african_american = apply_threshold(categorical_results['African-American'], thresholds['African-American'])
    caucasian = apply_threshold(categorical_results['Caucasian'], thresholds['Caucasian'])
    hispanic = apply_threshold(categorical_results['Hispanic'], thresholds['Hispanic'])
    other = apply_threshold(categorical_results['Other'], thresholds['Other'])

    # Storing the data.
    equal_opportunity_data['African-American'] = african_american
    equal_opportunity_data['Caucasian'] = caucasian
    equal_opportunity_data['Hispanic'] = hispanic
    equal_opportunity_data['Other'] = other

    return equal_opportunity_data, thresholds


def enforce_maximum_profit(categorical_results):
    """Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data."""

    mp_data = {}
    thresholds = {}

    # Function to calculate the accuracy for each group.
    def calculate_accuracy(group):
        accuracy_values = []
        for prediction in group:
            count = 0
            for x in range(len(prediction)):
                if prediction[x][0] == prediction[x][1]:
                    count += 1
            accuracy_values.append(count / len(prediction))
        return accuracy_values

    african_american_v = []
    caucasian_v = []
    hispanic_v = []
    other_v = []
    # Getting the prediction label pairs for all groups.
    for i in range(1, 101):
        threshold = float(i) / 100.0
        african_american = apply_threshold(categorical_results['African-American'], threshold)
        caucasian = apply_threshold(categorical_results['Caucasian'], threshold)
        hispanic = apply_threshold(categorical_results['Hispanic'], threshold)
        other = apply_threshold(categorical_results['Other'], threshold)
        african_american_v.append(african_american)
        caucasian_v.append(caucasian)
        hispanic_v.append(hispanic)
        other_v.append(other)

    # Getting the accuracies for all the possible threshold values.
    accuracy_values_african_american = calculate_accuracy(african_american_v)
    accuracy_values_caucasian = calculate_accuracy(caucasian_v)
    accuracy_values_hispanic = calculate_accuracy(hispanic_v)
    accuracy_values_other = calculate_accuracy(other_v)

    # Getting the maximum accuracies for each group.
    maximum_accuracy_african_american = max(accuracy_values_african_american)
    maximum_accuracy_caucasian = max(accuracy_values_caucasian)
    maximum_accuracy_hispanic = max(accuracy_values_hispanic)
    maximum_accuracy_other = max(accuracy_values_other)

    # Getting the final threshold values for the groups.
    # threshold_values = []
    for value in range(len(accuracy_values_african_american)):
        if accuracy_values_african_american[value] == maximum_accuracy_african_american:
            thresholds['African-American'] = (value + 1)/100
            break
    for value in range(len(accuracy_values_caucasian)):
        if accuracy_values_caucasian[value] == maximum_accuracy_caucasian:
            thresholds['Caucasian'] = (value + 1)/100
            break
    for value in range(len(accuracy_values_hispanic)):
        if accuracy_values_hispanic[value] == maximum_accuracy_hispanic:
            thresholds['Hispanic'] = (value + 1)/100
            break
    for value in range(len(accuracy_values_other)):
        if accuracy_values_other[value] == maximum_accuracy_other:
            thresholds['Other'] = (value + 1)/100
            break

    # Applying the thresholds on each group.
    african_american = apply_threshold(categorical_results['African-American'], thresholds['African-American'])
    caucasian = apply_threshold(categorical_results['Caucasian'], thresholds['Caucasian'])
    hispanic = apply_threshold(categorical_results['Hispanic'], thresholds['Hispanic'])
    other = apply_threshold(categorical_results['Other'], thresholds['Other'])

    # Storing the data.
    mp_data['African-American'] = african_american
    mp_data['Caucasian'] = caucasian
    mp_data['Hispanic'] = hispanic
    mp_data['Other'] = other

    # Must complete this function!
    return mp_data, thresholds


def enforce_predictive_parity(categorical_results, epsilon):
    """ Determine thresholds such that all groups have the same PPV, and return the best solution
        according to chosen secondary optimization criteria."""
    predictive_parity_data = {}
    thresholds = {}

    african_american_v = []
    caucasian_v = []
    hispanic_v = []
    other_v = []
    # Getting the prediction label pairs for all groups for all thresholds.
    for i in range(1, 101):
        threshold = float(i) / 100.0
        african_american = apply_threshold(categorical_results['African-American'], threshold)
        caucasian = apply_threshold(categorical_results['Caucasian'], threshold)
        hispanic = apply_threshold(categorical_results['Hispanic'], threshold)
        other = apply_threshold(categorical_results['Other'], threshold)
        african_american_v.append(african_american)
        caucasian_v.append(caucasian)
        hispanic_v.append(hispanic)
        other_v.append(other)

    # Getting PPV values for all the groups.
    african_american_ppv = []
    caucasian_ppv = []
    hispanic_ppv = []
    other_ppv = []
    for i in range(len(african_american_v)):
        pos_pred_value = get_positive_predictive_value(african_american_v[i])
        if pos_pred_value > 0:
            african_american_ppv.append(pos_pred_value)
    for i in range(len(caucasian_v)):
        pos_pred_value = get_positive_predictive_value(caucasian_v[i])
        if pos_pred_value > 0:
            caucasian_ppv.append(get_positive_predictive_value(caucasian_v[i]))
    for i in range(len(hispanic_v)):
        pos_pred_value = get_positive_predictive_value(hispanic_v[i])
        if pos_pred_value > 0:
            hispanic_ppv.append(pos_pred_value)
    for i in range(len(other_v)):
        pos_pred_value = get_positive_predictive_value(other_v[i])
        if pos_pred_value > 0:
            other_ppv.append(get_positive_predictive_value(other_v[i]))

    # Getting possible threshold values.
    possible_thresholds = []
    for i in range(len(african_american_ppv)):
        for j in range(len(caucasian_ppv)):
            a = compare_probability(african_american_ppv[i], caucasian_ppv[j], epsilon)
            if a is True:
                for k in range(len(hispanic_ppv)):
                    a = compare_probability(african_american_ppv[i], hispanic_ppv[k], epsilon)
                    b = compare_probability(caucasian_ppv[j], hispanic_ppv[k], epsilon)
                    if a and b is True:
                        for m in range(len(other_ppv)):
                            a = compare_probability(african_american_ppv[i], other_ppv[m], epsilon)
                            b = compare_probability(caucasian_ppv[j], other_ppv[m], epsilon)
                            c = compare_probability(hispanic_ppv[k], other_ppv[m], epsilon)
                            if a and b and c is True:
                                possible_thresholds.append([(i + 1)/100, (j + 1)/100, (k + 1)/100, (m + 1)/100])

    # Optimizing for accuracy as the secondary metric.
    # Getting the accuracies for all the possible threshold values.
    accuracy_values = []
    for i in range(len(possible_thresholds)):
        african_american = apply_threshold(categorical_results['African-American'], (possible_thresholds[i][0]))
        caucasian = apply_threshold(categorical_results['Caucasian'], (possible_thresholds[i][1]))
        hispanic = apply_threshold(categorical_results['Hispanic'], (possible_thresholds[i][2]))
        other = apply_threshold(categorical_results['Other'], (possible_thresholds[i][3]))
        predictive_parity_data['African-American'] = african_american
        predictive_parity_data['Caucasian'] = caucasian
        predictive_parity_data['Hispanic'] = hispanic
        predictive_parity_data['Other'] = other
        accuracy_values.append(get_total_accuracy(predictive_parity_data))

    # Finding the maximum accuracy.
    maximum = max(accuracy_values)

    # Getting the final threshold values for the groups.
    threshold_values = []
    for value in range(len(accuracy_values)):
        if accuracy_values[value] == maximum:
            threshold_values = value

    # Storing the thresholds in a dictionary.
    thresholds['African-American'] = (possible_thresholds[threshold_values][0])
    thresholds['Caucasian'] = (possible_thresholds[threshold_values][1])
    thresholds['Hispanic'] = (possible_thresholds[threshold_values][2])
    thresholds['Other'] = (possible_thresholds[threshold_values][3])

    # Applying the thresholds on each group.
    african_american = apply_threshold(categorical_results['African-American'], thresholds['African-American'])
    caucasian = apply_threshold(categorical_results['Caucasian'], thresholds['Caucasian'])
    hispanic = apply_threshold(categorical_results['Hispanic'], thresholds['Hispanic'])
    other = apply_threshold(categorical_results['Other'], thresholds['Other'])

    # Storing the data.
    predictive_parity_data['African-American'] = african_american
    predictive_parity_data['Caucasian'] = caucasian
    predictive_parity_data['Hispanic'] = hispanic
    predictive_parity_data['Other'] = other

    # Must complete this function!
    return predictive_parity_data, thresholds


def enforce_single_threshold(categorical_results):
    """ Apply a single threshold to all groups, and return the best solution according to
        chosen secondary optimization criteria."""
    single_threshold_data = {}
    thresholds = {}

    # Getting the accuracies for each threshold.
    accuracy_values = []
    for i in range(1, 101):
        threshold = float(i) / 100.0
        african_american = apply_threshold(categorical_results['African-American'], threshold)
        caucasian = apply_threshold(categorical_results['Caucasian'], threshold)
        hispanic = apply_threshold(categorical_results['Hispanic'], threshold)
        other = apply_threshold(categorical_results['Other'], threshold)
        single_threshold_data['African-American'] = african_american
        single_threshold_data['Caucasian'] = caucasian
        single_threshold_data['Hispanic'] = hispanic
        single_threshold_data['Other'] = other
        accuracy_values.append(get_total_accuracy(single_threshold_data))

    # Finding the maximum accuracy.
    maximum = max(accuracy_values)

    # Getting the final threshold values for the groups.
    threshold_values = []
    for value in range(len(accuracy_values)):
        if accuracy_values[value] == maximum:
            threshold_values = (value + 1)/100

    # Storing the thresholds in a dictionary.
    thresholds['African-American'] = threshold_values
    thresholds['Caucasian'] = threshold_values
    thresholds['Hispanic'] = threshold_values
    thresholds['Other'] = threshold_values

    # Applying the thresholds on each group.
    african_american = apply_threshold(categorical_results['African-American'], thresholds['African-American'])
    caucasian = apply_threshold(categorical_results['Caucasian'], thresholds['Caucasian'])
    hispanic = apply_threshold(categorical_results['Hispanic'], thresholds['Hispanic'])
    other = apply_threshold(categorical_results['Other'], thresholds['Other'])

    # Storing the data.
    single_threshold_data['African-American'] = african_american
    single_threshold_data['Caucasian'] = caucasian
    single_threshold_data['Hispanic'] = hispanic
    single_threshold_data['Other'] = other

    # Must complete this function!
    return single_threshold_data, thresholds

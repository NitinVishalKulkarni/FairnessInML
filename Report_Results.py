from Postprocessing import *
from utils import *
from datetime import datetime
import copy


def report_results(data):

    begin = datetime.now()

    print("Attempting to enforce demographic parity...")
    demographic_parity_data, demographic_parity_thresholds = enforce_demographic_parity(copy.deepcopy(data), 0.02)
    if demographic_parity_data is not None:

        print("--------------------DEMOGRAPHIC PARITY RESULTS--------------------")
        print("")
        for group in demographic_parity_data.keys():
            num_positive_predictions = get_num_predicted_positives(demographic_parity_data[group])
            prob = num_positive_predictions / len(demographic_parity_data[group])
            print("Probability of positive prediction for " + str(group) + ": " + str(prob))


        print("")
        for group in demographic_parity_data.keys():
            accuracy = get_num_correct(demographic_parity_data[group]) / len(demographic_parity_data[group])
            print("Accuracy for " + group + ": " + str(accuracy))

        print("")
        for group in demographic_parity_data.keys():
            FPR = get_false_positive_rate(demographic_parity_data[group])
            print("FPR for " + group + ": " + str(FPR))

        print("")
        for group in demographic_parity_data.keys():
            FNR = get_false_negative_rate(demographic_parity_data[group])
            print("FNR for " + group + ": " + str(FNR))

        print("")
        for group in demographic_parity_data.keys():
            TPR = get_true_positive_rate(demographic_parity_data[group])
            print("TPR for " + group + ": " + str(TPR))

        print("")
        for group in demographic_parity_data.keys():
            TNR = get_true_negative_rate(demographic_parity_data[group])
            print("TNR for " + group + ": " + str(TNR))

        print("")
        for group in demographic_parity_thresholds.keys():
            print("Threshold for " + group + ": " + str(demographic_parity_thresholds[group]))

        print("")
        total_cost = apply_financials(demographic_parity_data)
        print("Total cost: ")
        print('${:,.0f}'.format(total_cost))
        total_accuracy = get_total_accuracy(demographic_parity_data)
        print("Total accuracy: " + str(total_accuracy))
        print("-----------------------------------------------------------------")
        print("")

    print("Attempting to enforce equal opportunity...")
    equal_opportunity_data, equal_opportunity_thresholds = enforce_equal_opportunity(copy.deepcopy(data), 0.01)
    if equal_opportunity_data is not None:
        print("--------------------EQUAL OPPORTUNITY RESULTS--------------------")
        print("")
        for group in equal_opportunity_data.keys():
            accuracy = get_num_correct(equal_opportunity_data[group]) / len(equal_opportunity_data[group])
            print("Accuracy for " + group + ": " + str(accuracy))

        print("")
        for group in equal_opportunity_data.keys():
            FPR = get_false_positive_rate(equal_opportunity_data[group])
            print("FPR for " + group + ": " + str(FPR))

        print("")
        for group in equal_opportunity_data.keys():
            FNR = get_false_negative_rate(equal_opportunity_data[group])
            print("FNR for " + group + ": " + str(FNR))

        print("")
        for group in equal_opportunity_data.keys():
            TPR = get_true_positive_rate(equal_opportunity_data[group])
            print("TPR for " + group + ": " + str(TPR))

        print("")
        for group in equal_opportunity_data.keys():
            TNR = get_true_negative_rate(equal_opportunity_data[group])
            print("TNR for " + group + ": " + str(TNR))

        print("")
        for group in equal_opportunity_thresholds.keys():
            print("Threshold for " + group + ": " + str(equal_opportunity_thresholds[group]))

        print("")
        total_cost = apply_financials(equal_opportunity_data)
        print("Total cost: ")
        print('${:,.0f}'.format(total_cost))
        total_accuracy = get_total_accuracy(equal_opportunity_data)
        print("Total accuracy: " + str(total_accuracy))
        print("-----------------------------------------------------------------")
        print("")


    print("Attempting to enforce maximum profit...")
    max_profit_data, max_profit_thresholds = enforce_maximum_profit(copy.deepcopy(data))
    if max_profit_data is not None:
        print("--------------------MAXIMUM PROFIT RESULTS--------------------")
        print("")
        for group in max_profit_data.keys():
            accuracy = get_num_correct(max_profit_data[group]) / len(max_profit_data[group])
            print("Accuracy for " + group + ": " + str(accuracy))

        print("")
        for group in max_profit_data.keys():
            FPR = get_false_positive_rate(max_profit_data[group])
            print("FPR for " + group + ": " + str(FPR))

        print("")
        for group in max_profit_data.keys():
            FNR = get_false_negative_rate(max_profit_data[group])
            print("FNR for " + group + ": " + str(FNR))

        print("")
        for group in max_profit_data.keys():
            TPR = get_true_positive_rate(max_profit_data[group])
            print("TPR for " + group + ": " + str(TPR))

        print("")
        for group in max_profit_data.keys():
            TNR = get_true_negative_rate(max_profit_data[group])
            print("TNR for " + group + ": " + str(TNR))

        print("")
        for group in max_profit_thresholds.keys():
            print("Threshold for " + group + ": " + str(max_profit_thresholds[group]))

        print("")
        total_cost = apply_financials(max_profit_data)
        print("Total cost: ")
        print('${:,.0f}'.format(total_cost))
        total_accuracy = get_total_accuracy(max_profit_data)
        print("Total accuracy: " + str(total_accuracy))

        print("-----------------------------------------------------------------")
        print("")

    print("Attempting to enforce predictive parity...")
    predictive_parity_data, predictive_parity_thresholds = enforce_predictive_parity(copy.deepcopy(data), 0.01)
    if predictive_parity_data is not None:
        print("--------------------PREDICTIVE PARITY RESULTS--------------------")
        print("")
        for group in predictive_parity_data.keys():
            accuracy = get_num_correct(predictive_parity_data[group]) / len(predictive_parity_data[group])
            print("Accuracy for " + group + ": " + str(accuracy))

        print("")
        for group in predictive_parity_data.keys():
            PPV = get_positive_predictive_value(predictive_parity_data[group])
            print("PPV for " + group + ": " + str(PPV))

        print("")
        for group in predictive_parity_data.keys():
            FPR = get_false_positive_rate(predictive_parity_data[group])
            print("FPR for " + group + ": " + str(FPR))

        print("")
        for group in predictive_parity_data.keys():
            FNR = get_false_negative_rate(predictive_parity_data[group])
            print("FNR for " + group + ": " + str(FNR))

        print("")
        for group in predictive_parity_data.keys():
            TPR = get_true_positive_rate(predictive_parity_data[group])
            print("TPR for " + group + ": " + str(TPR))

        print("")
        for group in predictive_parity_data.keys():
            TNR = get_true_negative_rate(predictive_parity_data[group])
            print("TNR for " + group + ": " + str(TNR))

        print("")
        for group in predictive_parity_thresholds.keys():
            print("Threshold for " + group + ": " + str(predictive_parity_thresholds[group]))

        print("")
        total_cost = apply_financials(predictive_parity_data)
        print("Total cost: ")
        print('${:,.0f}'.format(total_cost))
        total_accuracy = get_total_accuracy(predictive_parity_data)
        print("Total accuracy: " + str(total_accuracy))
        print("-----------------------------------------------------------------")
        print("")

    print("Attempting to enforce single threshold...")
    single_threshold_data, single_thresholds = enforce_single_threshold(copy.deepcopy(data))
    if single_threshold_data is not None:
        print("--------------------SINGLE THRESHOLD RESULTS--------------------")
        print("")
        for group in single_threshold_data.keys():
            accuracy = get_num_correct(single_threshold_data[group]) / len(single_threshold_data[group])
            print("Accuracy for " + group + ": " + str(accuracy))

        print("")
        for group in single_threshold_data.keys():
            FPR = get_false_positive_rate(single_threshold_data[group])
            print("FPR for " + group + ": " + str(FPR))

        print("")
        for group in single_threshold_data.keys():
            FNR = get_false_negative_rate(single_threshold_data[group])
            print("FNR for " + group + ": " + str(FNR))

        print("")
        for group in single_threshold_data.keys():
            TPR = get_true_positive_rate(single_threshold_data[group])
            print("TPR for " + group + ": " + str(TPR))

        print("")
        for group in single_threshold_data.keys():
            TNR = get_true_negative_rate(single_threshold_data[group])
            print("TNR for " + group + ": " + str(TNR))

        print("")
        for group in single_thresholds.keys():
            print("Threshold for " + group + ": " + str(single_thresholds[group]))

        print("")
        total_cost = apply_financials(single_threshold_data)
        print("Total cost: ")
        print('${:,.0f}'.format(total_cost))
        total_accuracy = get_total_accuracy(single_threshold_data)
        print("Total accuracy: " + str(total_accuracy))
        print("-----------------------------------------------------------------")

        end = datetime.now()

        seconds = end-begin
        print("Postprocessing took approximately: " + str(seconds) + " seconds")



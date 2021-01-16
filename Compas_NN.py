import numpy as np
from keras import regularizers
from keras.layers import Dense
from keras.models import Sequential
from Preprocessing import preprocess
from Report_Results import report_results
from utils import *
import time
start = time.time()


def neural_network_classification(metrics_):
    """This function takes as input  a list of metrics and uses a neural network to perform the classification.
    :param metrics_: List containing the metrics that we want to use for classification."""

    training_data, training_labels, test_data, test_labels, categories_, mappings_ = preprocess(metrics_)

    activation = "relu"
    model = Sequential()
    model.add(Dense(len(metrics)*2, activation=activation, kernel_regularizer=regularizers.l2(0.1),
                    input_shape=(len(metrics))))
    model.add(Dense(30, activation=activation, kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss="binary_crossentropy")
    model.fit(training_data, training_labels, epochs=30, batch_size=300, validation_data=(test_data, test_labels),
              verbose=1)

    data_ = np.concatenate((training_data, test_data))
    labels_ = np.concatenate((training_labels, test_labels))

    predictions_ = model.predict(data_)
    predictions_ = np.squeeze(predictions_, axis=1)
    print("End")
    return data, predictions_, labels_, categories_, mappings_


metrics = ["sex", "age_cat", "race", 'c_charge_degree', 'priors_count']

# Changing the int value sets the number of models to create before choosing the "best" one
data, predictions, labels, categories, mappings = neural_network_classification(metrics)
race_cases = get_cases_by_metric(data, categories, "race", mappings, predictions, labels)
print('This is taking a long time:')
report_results(race_cases)
end = time.time()
print(start - end)

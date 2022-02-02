#    See info_gain_notes.md if you need an explanation of the math.

import math  # for log computation


def calculate_entropy(x, y, classes):
    """
    This function calculates the entropy for a given dataset.
    It's equivalent to formula (2) in the CW spec pg 6.
    Args:
        x: a 2D numpy matrice consisting of all the instances in the dataset
            and their corresponding attribute values
        y: a 1D numpy array consisting of the class labels of all instances
        classes: a 1D numpy array consisting of the
    Returns:
        float: the entropy for the given dataset.
    """
    number_of_instances = len(y)
    label_frequencies = dict.fromkeys(classes, 0)
    for label in y:
        label_frequencies[label] += 1

    entropy = 0.0
    for label, value in label_frequencies.items():
        if value != 0:
            probability = value / number_of_instances
            if probability > 0 and probability < 1:
                entropy -= probability * math.log(probability, 2)

    # TODO: add sanity check

    return entropy


def calculate_best_info_gain(x, y, classes):
    """
    Args:
    - As with the `calculate_entropy` function, but x, y and classes
        for the parent dataset
    Returns:
    - Index of attribute with highest information gain.
    Assume:
    - We treat the attribute values as integers.
    - We will have 3 bins: when attribute value x is:
        . less than 5
        . between 5 and 10 (inclusive)
        . more than 10
     # TODO: consider alternative way to split: this feels like it'll overfit.
        e.g. it wouldn't really work for datasets that don't fall within those ranges.
        Another way would be to work with `max` and `min`.
    """
    dataset_entropy = calculate_entropy(x, y, classes)
    number_of_attributes = len(x[0, :])
    number_of_instances = len(y)
    # for each attribute in the dataset
    for attribute in range(number_of_attributes):
        # split instances into three bins
        bin_1_filter = x[:, attribute] < 5
        bin_2_filter = x[:, attribute] >= 5 and x[:, attribute] <= 10
        bin_3_filter = x[:, attribute] >= 10
        x_bin_1 = x[:, attribute][bin_1_filter]
        y_bin_1 = y[bin_1_filter]
        x_bin_2 = x[:, attribute][bin_2_filter]
        y_bin_2 = y[bin_2_filter]
        x_bin_3 = x[:, attribute][bin_3_filter]
        y_bin_3 = y[bin_3_filter]
        # calculate the entropy for each
        # DELIBERATELY INCOMPLETE: I don't think we should use this method.


# Main Function, follows pseudocode given in spec.
def induce_decision_tree():
    """
    This function computes the information gain for different subsets of
    the dataset passed in for different attribute and split points.
    Input:
    """
    pass

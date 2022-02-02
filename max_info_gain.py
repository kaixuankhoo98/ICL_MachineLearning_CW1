#    See info_gain_notes.md if you need an explanation of the math.

import math  # for log computation
import numpy as np


def calculate_entrophy(x, y, classes):
    """
    This function calculates the entrophy for a given dataset.
    It's equivalent to formula (2) in the CW spec pg 6.
    Args:
        x: a 2D numpy matrice consisting of all the instances in the dataset
            and their corresponding attribute values
        y: a 1D numpy array consisting of the class labels of all instances
        classes: a 1D numpy array consisting of the
    Returns:
        float: the entrophy for the given dataset.
    """
    number_of_instances = len(y)
    label_frequencies = dict.fromkeys(classes, 0)
    for label in y:
        label_frequencies[label] += 1

    entrophy = 0.0
    for label, value in label_frequencies.items():
        if value != 0:
            probability = value / number_of_instances
            if probability > 0 and probability < 1:
                entrophy -= probability * math.log(probability, 2)

    # TODO: add sanity check

    return entrophy


def calculate_best_info_gain(x, y, classes):
    """
    Args:
    - As with the `calculate_entrophy` function, but x, y and classes
        for a parent dataset
    Returns a tuple with:
    - Index of attribute with highest information gain,
    - and the value of that attribute to split along for max info gain.
    Assume:
    - We treat the attribute values as integers.
    """
    dataset_entrophy = calculate_entrophy(x, y, classes)
    number_of_attributes = len(x[0, :])
    number_of_instances = len(y)
    container = []  # to store info on each split done below.
    # for each attribute in the dataset
    for attribute_index in range(number_of_attributes):
        attribute_max_value = x[:, attribute_index].max()
        attribute_min_value = x[:, attribute_index].min()
        # split the dataset two ways along each of the values of those attributes
        # and caluclate the respective entrophies
        for attribute_value in range(
            attribute_min_value, attribute_max_value + 1
        ):  # +1 to include max
            # CALCULATE ENTROPHY FOR LEFT
            left_filter = x[:, attribute_index] >= attribute_value
            left_filtered_x = x[left_filter, :]
            left_filtered_y = y[left_filter]
            left_entrophy = calculate_entrophy(
                left_filtered_x, left_filtered_y, classes
            )
            # CALCULATE ENTROPHY FOR RIGHT
            right_filter = np.invert(left_filter)
            right_filtered_x = x[right_filter, :]
            right_filtered_y = y[right_filter]
            right_entrophy = calculate_entrophy(
                right_filtered_x, right_filtered_y, classes
            )
            # CALUCLATE INFORMATION GAIN FOR splitting ALONG THIS ATTRIBUTE
            # AND THIS PARTICULAR VALUE
            proportion = len(left_filtered_y) / (
                len(left_filtered_y) + len(left_filtered_x)
            )
            info_gained = dataset_entrophy - (
                proportion * left_entrophy + (1 - proportion) * right_entrophy
            )
            # Save info into container
            container.append([info_gained, attribute_index, attribute_value])
    # find attribute_index and attribute_value which results in the split with
    # the greatest information gain
    max_info_gained = 0.0
    best_attribute_index = 0
    best_attribute_value = 0
    for a_list in container:
        if max_info_gained < a_list[0]:  # 0 is index of each splits info_gain
            max_info_gained = a_list[0]
            best_attribute_index = a_list[1]
            best_attribute_value = a_list[2]
    print(max_info_gained, best_attribute_index, best_attribute_value)
    return (best_attribute_index, best_attribute_value)


# Main Function, follows pseudocode given in spec.
def induce_decision_tree():
    """
    This function computes the information gain for different subsets of
    the dataset passed in for different attribute and split points.
    Input:
    """
    pass

#    See info_gain_notes.md if you need an explanation of the math.

import math  # for log computation
import numpy as np
from node import Node


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
        for label2 in classes:
            if label2 == label:
                label_frequencies[label2] += 1
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
        for a parent dataset
    Returns a tuple with:
    - Index of attribute with highest information gain,
    - and the value of that attribute to split along for max info gain.
    Assume:
    - We treat the attribute values as integers.
    """
    dataset_entropy = calculate_entropy(x, y, classes)
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
            # CALCULATE entropy FOR LEFT
            left_filter = x[:, attribute_index] <= attribute_value
            left_filtered_x = x[left_filter, :]
            left_filtered_y = y[left_filter]
            left_entropy = calculate_entropy(left_filtered_x, left_filtered_y, classes)
            # CALCULATE entropy FOR RIGHT
            right_filter = np.invert(left_filter)
            right_filtered_x = x[right_filter, :]
            right_filtered_y = y[right_filter]
            right_entropy = calculate_entropy(
                right_filtered_x, right_filtered_y, classes
            )
            # CALUCLATE INFORMATION GAIN FOR splitting ALONG THIS ATTRIBUTE
            # AND THIS PARTICULAR VALUE
            proportion = len(left_filtered_y) / number_of_instances
            info_gained = dataset_entropy - (
                proportion * left_entropy + (1 - proportion) * right_entropy
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


def find_best_node(x, y, classes):
    best_attribute_index, best_attribute_value = calculate_best_info_gain(x, y, classes)

    best_node = Node(best_attribute_index, best_attribute_value, x, y, classes)

    return best_node


# helper function for induce_decision_tree
def same_labels(y):
    # TODO: think about optimizing using an ndarry method instead.
    test = y[0]
    for i in y:
        if test != i:
            return False
    return True


# helper function for induce_decision_tree
def split_dataset(x, y, classes, best_node):
    attribute_to_split_by = best_node.attribute_index
    attribute_value_to_split_by = best_node.attribute_value
    left_filter = x[:, attribute_to_split_by] <= attribute_value_to_split_by
    left_child_x = x[left_filter]
    left_child_y = y[left_filter]
    left_child_classes = np.unique(left_child_y)
    right_filter = np.invert(left_filter)
    right_child_x = x[right_filter]
    right_child_y = y[right_filter]
    right_child_classes = np.unique(right_child_y)
    return (
        left_child_x,
        left_child_y,
        left_child_classes,
        right_child_x,
        right_child_y,
        right_child_classes,
    )


def no_more_splits(x):
    if len(x) < 2:
        return True
    else:
        return False


# Main Function, follows pseudocode given in spec.
def induce_decision_tree(x, y, classes):
    """

    Input:
    """
    # BASE CASES: all instances in
    if len(x) < 2 or same_labels(y):
        leaf_node = Node(None, None, x, y, classes)
        leaf_node.last_node = True
        return leaf_node
    else:
        best_node = find_best_node(x, y, classes)
        left_x, left_y, left_classes, right_x, right_y, right_classes = split_dataset(
            x, y, classes, best_node
        )
        child_node_left = induce_decision_tree(left_x, left_y, left_classes)
        best_node.add_child(child_node_left)
        child_node_right = induce_decision_tree(right_x, right_y, right_classes)
        best_node.add_child(child_node_right)
        return best_node


def find_majority_label(y):
    """
    Input:
    y is a numpy array of size (K, ) containing the class label for each instance in a dataset.

    Returns:
    the majority class label in y
    """
    vals, counts = np.unique(y, return_counts=True)
    index = np.argmax(counts)
    return vals[index]

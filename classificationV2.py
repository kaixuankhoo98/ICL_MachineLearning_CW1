#############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit() and predict() methods of DecisionTreeClassifier.
# You are free to add any other methods as needed.
##############################################################################

import numpy as np
import max_info_gain as mig
from node import Node


class DecisionTreeClassifier(object):
    """Basic decision tree classifier

    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained

    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self):
        self.is_trained = False
        DecisionTree = Node(attribute_index=None, attribute_value=None, x=None, y=None, classes=None)

    def traverse_and_print_tree(self):
        self.DecisionTree.print_tree()

    def fit(self, x, y):
        """Constructs a decision tree classifier from data

        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K)
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str
        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(
            y
        ), "Training failed. x and y must have the same number of instances."

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################

        # convert y into y_index and classes to pass into max info gain functions
        [classes, y_index] = np.unique(y, return_inverse=True)
        mig.calculate_best_info_gain(x, y_index, classes)
        self.DecisionTree = mig.induce_decision_tree(x, y_index, classes)
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

    def prune_tree(self):
        if not self.is_trained:
            raise Exception("Tree has not been constructed.")
        node_iterator = self.DecisionTree
        #TODO: implement iterating through the tree.
        pass

    def predict(self, x):
        """Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K)
                           M is the number of test instances
                           K is the number of attributes

        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x
        """

        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")

        # set up an empty (M, ) numpy array to store the predicted labels
        # feel free to change this if needed
        predictions = []

        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        for test_instance in x:
            node_pointer = self.DecisionTree
            while(node_pointer.last_node == False):
                if(test_instance[node_pointer.attribute_index] <= node_pointer.attribute_value):
                    node_pointer = node_pointer.left_child
                else:
                    node_pointer = node_pointer.right_child
            predictions.append(mig.find_majority_label(node_pointer.y))

        predictions = np.array(predictions)

        return predictions

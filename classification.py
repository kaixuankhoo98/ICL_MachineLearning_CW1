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

        print("Trying info_gain function...")
        mig.calculate_best_info_gain(x, y_index, classes)
        print("Now, trying induce decision tree...")
        self.DecisionTree = mig.induce_decision_tree(x, y_index, classes)
        print("First node values are:")
        print(self.DecisionTree.attribute_index)
        print(self.DecisionTree.attribute_value)
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

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
        predictions = np.zeros((x.shape[0],), dtype=np.object)

        predictions2 = []

        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        for test_instance in x:
            temp_node = self.DecisionTree
            count = 0
            for value in test_instance:
                print(temp_node.y[0])
                if count == temp_node.attribute_index:
                    print("here")
                    if temp_node.last_node == False:
                        if value <= temp_node.attribute_value:
                            temp_node = temp_node.left_child
                        else:
                            temp_node = temp_node.right_child
                count += 1
            predictions2.append(temp_node.y[0])
            

        # for test_instance in x:
        #     for attribute_value in test_instance:
        #         count = 0
        #         while (
        #             self.DecisionTree.left_child != None
        #             or self.DecisionTree.right_child != None
        #         ):
        #             count += 1
        #             print(temp_node.attribute_index)
        #             if count == temp_node.attribute_index:
        #                 if attribute_value <= temp_node.attribute_value:
        #                     ...
        #                     temp_node = temp_node.left_child
        #                 else:
        #                     temp_node = temp_node.right_child
        #         predictions.append(temp_node.y)
        # remember to change this if you rename the variable
        return predictions2

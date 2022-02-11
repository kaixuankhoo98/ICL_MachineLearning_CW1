##############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train_and_predict() function. 
#             You are free to add any other methods as needed. 
##############################################################################

import numpy as np
from classificationV2 import ImprovedDecisionTreeClassifier
from evaluation import accuracy


def train_and_predict(x_train, y_train, x_test, x_val, y_val):
    """ Interface to train and test the new/improved decision tree.
    
    This function is an interface for training and testing the new/improved
    decision tree classifier. 

    x_train and y_train should be used to train your classifier, while 
    x_test should be used to test your classifier. 
    x_val and y_val may optionally be used as the validation dataset. 
    You can just ignore x_val and y_val if you do not need a validation dataset.

    Args:
    x_train (numpy.ndarray): Training instances, numpy array of shape (N, K) 
                       N is the number of instances
                       K is the number of attributes
    y_train (numpy.ndarray): Class labels, numpy array of shape (N, )
                       Each element in y is a str 
    x_test (numpy.ndarray): Test instances, numpy array of shape (M, K) 
                            M is the number of test instances
                            K is the number of attributes
    x_val (numpy.ndarray): Validation instances, numpy array of shape (L, K) 
                       L is the number of validation instances
                       K is the number of attributes
    y_val (numpy.ndarray): Class labels of validation set, numpy array of shape (L, )
    """

    #######################################################################
    #                 ** TASK 4.1: COMPLETE THIS FUNCTION **
    #######################################################################
       

    # Train new classifier

    improved_classifier = ImprovedDecisionTreeClassifier()
    improved_classifier.fit_hyper(x_train, y_train, 4)
    

    # set up an empty (M, ) numpy array to store the predicted labels 
    # feel free to change this if needed
    # predictions = np.zeros((x_test.shape[0],), dtype=np.object)
        
    # Make predictions on x_test using new classifier 
    improved_predictions = improved_classifier.predict(x_val)
    improved_accuracy = accuracy(y_val, improved_predictions)
    print("The accuracy of improved classifier prior to pruning is ", improved_accuracy)
    print("Now, trying to prune the improved classifier")
    pruned_accuracy = improved_classifier.test_pruning_tree(improved_classifier.DecisionTree, x_val, y_val, improved_accuracy)
    print("The accuracy of the pruned classifier is ", pruned_accuracy)
    print("Predicting with pruned tree now:")
    improved_predictions = improved_classifier.predict(x_val)
    improved_predictions = np.array(improved_predictions)     
        
    return improved_predictions


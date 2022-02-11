from calendar import c
import read_data as rd
import numpy as np
import statistics as stats
import classification as model
import cross_validation as cv
import classificationV2 as model2
from numpy.random import default_rng

def accuracy(y_gold, y_prediction):
    """ Compute the accuracy given the ground truth and predictions

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        float : the accuracy
    """

    assert len(y_gold) == len(y_prediction)  
    
    try:
        return np.sum(y_gold == y_prediction) / len(y_gold)
    except ZeroDivisionError:
        return 0.


def confusion_matrix(y_gold, y_prediction, class_labels=None):
    """ Compute the confusion matrix.
        
    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels
        class_labels (np.ndarray): a list of unique class labels. 
                               Defaults to the union of y_gold and y_prediction.

    Returns:
        np.array : shape (C, C), where C is the number of classes. 
                   Rows are ground truth per class, columns are predictions
    """

    # if no class_labels are given, we obtain the set of unique class labels from
    # the union of the ground truth annotation and the prediction
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

    # for each correct class (row), 
    # compute how many instances are predicted for each class (columns)
    for (i, label) in enumerate(class_labels):
        # get predictions where the ground truth is the current class label
        indices = (y_gold == label)
        gold = y_gold[indices]
        predictions = y_prediction[indices]

        # quick way to get the counts per label
        (unique_labels, counts) = np.unique(predictions, return_counts=True)

        # convert the counts to a dictionary
        frequency_dict = dict(zip(unique_labels, counts))

        # fill up the confusion matrix for the current row
        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion


def accuracy_from_confusion(confusion):
    """ Compute the accuracy given the confusion matrix

    Args:
        confusion (np.ndarray): shape (C, C), where C is the number of classes. 
                    Rows are ground truth per class, columns are predictions

    Returns:
        float : the accuracy
    """   

    if np.sum(confusion) > 0:
        return np.sum(np.diag(confusion)) / np.sum(confusion)
    else:
        return 0.


def precision(y_gold, y_prediction):
    """ Compute the precision score per class given the ground truth and predictions
        
    Also return the macro-averaged precision across classes.
        
    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (precisions, macro_precision) where
            - precisions is a np.ndarray of shape (C,), where each element is the 
              precision for class c
            - macro-precision is macro-averaged precision (a float) 
    """

    confusion = confusion_matrix(y_gold, y_prediction)
    p = np.zeros((len(confusion), ))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[:, c]) > 0:
            p[c] = confusion[c, c] / np.sum(confusion[:, c])

    ## Alternative solution without computing the confusion matrix
    #class_labels = np.unique(np.concatenate((y_gold, y_prediction)))
    #p = np.zeros((len(class_labels), ))
    #for (c, label) in enumerate(class_labels):
    #    indices = (y_prediction == label) # get instances predicted as label
    #    correct = np.sum(y_gold[indices] == y_prediction[indices]) # intersection
    #    if np.sum(indices) > 0:
    #        p[c] = correct / np.sum(indices)     

    # Compute the macro-averaged precision
    macro_p = 0.
    if len(p) > 0:
        macro_p = np.mean(p)
    
    return (p, macro_p)



def recall(y_gold, y_prediction):
    """ Compute the recall score per class given the ground truth and predictions
        
    Also return the macro-averaged recall across classes.
        
    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (recalls, macro_recall) where
            - recalls is a np.ndarray of shape (C,), where each element is the 
                recall for class c
            - macro-recall is macro-averaged recall (a float) 
    """

    confusion = confusion_matrix(y_gold, y_prediction)
    r = np.zeros((len(confusion), ))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[c, :]) > 0:
            r[c] = confusion[c, c] / np.sum(confusion[c, :])

    ## Alternative solution without computing the confusion matrix
    #class_labels = np.unique(np.concatenate((y_gold, y_prediction)))
    #r = np.zeros((len(class_labels), ))
    #for (c, label) in enumerate(class_labels):
    #    indices = (y_gold == label) # get instances for current label
    #    correct = np.sum(y_gold[indices] == y_prediction[indices]) # intersection
    #    if np.sum(indices) > 0:
    #        r[c] = correct / np.sum(indices)     

    # Compute the macro-averaged recall
    macro_r = 0.
    if len(r) > 0:
        macro_r = np.mean(r)
    
    return (r, macro_r)


def f1_score(y_gold, y_prediction):
    """ Compute the F1-score per class given the ground truth and predictions
        
    Also return the macro-averaged F1-score across classes.
        
    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (f1s, macro_f1) where
            - f1s is a np.ndarray of shape (C,), where each element is the 
              f1-score for class c
            - macro-f1 is macro-averaged f1-score (a float) 
    """

    (precisions, macro_p) = precision(y_gold, y_prediction)
    (recalls, macro_r) = recall(y_gold, y_prediction)

    # just to make sure they are of the same length
    assert len(precisions) == len(recalls)

    f = np.zeros((len(precisions), ))
    for c, (p, r) in enumerate(zip(precisions, recalls)):
        if p + r > 0:
            f[c] = 2 * p * r / (p + r)

    # Compute the macro-averaged F1
    macro_f = 0.
    if len(f) > 0:
        macro_f = np.mean(f)
    
    return (f, macro_f)

def evaluate ( train_filepath, test_filepath ):

    # Training Data
    (x_train, y_train, classes_train) = rd.read_dataset( train_filepath )
    y_train_letters = []
    for i in y_train:
        y_train_letters.append(classes_train[i])
    y_train_letters = np.array(y_train_letters)

    # Test Data
    (x_test, y_test, classes_test) = rd.read_dataset(test_filepath)

    # Fit Decision Tree
    classifier = model.DecisionTreeClassifier()
    classifier.fit(x_train, y_train_letters)

    # Predicting
    y_pred=classifier.predict(x_test)

    # Print metrics
    print("Trained on: ", train_filepath)
    print("Tested on: ", test_filepath, '\n')
    print("Accuracy:")
    print(accuracy(y_test,y_pred))
    print('\n', "Confusion matrix:")
    print(confusion_matrix(y_test,y_pred))
    print('\n', "Recall:")
    print(recall(y_test,y_pred))
    print('\n', "Precision:")
    print(precision(y_test,y_pred))
    print('\n', 'F1 Score: ')
    print(f1_score(y_test,y_pred))

    return

def cross_validation( train_filepath, n_folds ):

    # Training Data
    (x, y, classes) = rd.read_dataset( train_filepath )
    y_letters = []
    for i in y:
        y_letters.append(classes[i])
    y_letters = np.array(y_letters)

    # Set randomness of cross validation
    seed = 60012
    rg = default_rng(seed)

    accuracies = np.zeros((n_folds, ))
    trees = []

    for i, (train_indices, test_indices) in enumerate(cv.train_test_k_fold(n_folds, len(x), rg)):
        # Get the dataset into correct splits
        x_train = x[train_indices, :]
        y_train = y[train_indices]
        y_train_letters = y_letters[train_indices]
        x_test = x[test_indices, :]
        y_test = y[test_indices]
        y_test_letters = y_letters[test_indices]

        # Train decision tree on data sets
        classifier = model.DecisionTreeClassifier()
        classifier.fit(x_train, y_train_letters)
        trees.append(classifier)

        # Predict
        predictions = classifier.predict(x_test)

        # Compute accuracy & add to 'accuracies' array
        acc = accuracy(y_test, predictions)
        accuracies[i] = acc

    print("Accuracy array: ", accuracies, '\n')
    print("Accuracy mean: ", accuracies.mean(), '\n')
    print("Standard Deviation: ", accuracies.std(), '\n') 

    # Returns trained trees
    return trees

    # # SHORTCOMING IS THAT THERE ARE SOMETIMES CLASSES THAT ARE AS COMMON AS ONE ANOTHER, PYTHON 3 STATS MODULE PICKS THE FIRST MODAL CLASS
def modal_cross_validation( test_filepath, classifiers ):
    
    # Test Data
    (x_test, y_test, classes_test) = rd.read_dataset(test_filepath)

    all_predictions = []
    for i in range( len(classifiers) ):
        all_predictions.append(classifiers[i].predict(x_test))

    # Iterate through all predictions
    modal_labels = []
    for pred in range ( len( all_predictions[0] ) ):
        list = []
        # Iterate through each tree
        for tree in range ( len ( all_predictions ) ):
            list.append( all_predictions[ tree ][ pred ] )
        modal_labels.append( stats.mode(list) )

    modal_accuracy = accuracy(y_test, modal_labels)

    return modal_accuracy


def cross_validation_improved( train_filepath, n_folds ):

    # Training Data
    (x, y, classes) = rd.read_dataset( train_filepath )
    y_letters = []
    for i in y:
        y_letters.append(classes[i])
    y_letters = np.array(y_letters)

    # Set randomness of cross validation
    seed = 60012
    rg = default_rng(seed)

    accuracies = np.zeros((n_folds, ))
    trees = []

    for i, (train_indices, test_indices) in enumerate(cv.train_test_k_fold(n_folds, len(x), rg)):
        # Get the dataset into correct splits
        x_train = x[train_indices, :]
        y_train = y[train_indices]
        y_train_letters = y_letters[train_indices]
        x_test = x[test_indices, :]
        y_test = y[test_indices]
        y_test_letters = y_letters[test_indices]

        # Train decision tree on data sets
        classifier = model.DecisionTreeClassifier()
        classifier.fit_hyper(x_train, y_train_letters, 4)
        trees.append(classifier)

        # Predict
        predictions = classifier.predict(x_test)

        # Compute accuracy & add to 'accuracies' array
        acc = accuracy(y_test, predictions)
        accuracies[i] = acc

    print("Accuracy array: ", accuracies, '\n')
    print("Accuracy mean: ", accuracies.mean(), '\n')
    print("Standard Deviation: ", accuracies.std(), '\n') 

    # Returns trained trees
    return trees

    # # SHORTCOMING IS THAT THERE ARE SOMETIMES CLASSES THAT ARE AS COMMON AS ONE ANOTHER, PYTHON 3 STATS MODULE PICKS THE FIRST MODAL CLASS
def modal_cross_validation_improved( test_filepath, classifiers ):
    
    # Test Data
    (x_test, y_test, classes_test) = rd.read_dataset(test_filepath)

    all_predictions = []
    for i in range( len(classifiers) ):
        all_predictions.append(classifiers[i].predict(x_test))

    # Iterate through all predictions
    modal_labels = []
    for pred in range ( len( all_predictions[0] ) ):
        list = []
        # Iterate through each tree
        for tree in range ( len ( all_predictions ) ):
            list.append( all_predictions[ tree ][ pred ] )
        modal_labels.append( stats.mode(list) )

    modal_accuracy = accuracy(y_test, modal_labels)

    return modal_accuracy

    
# # 3.1:

# evaluate( 'data/train_full.txt', 'data/test.txt' )
# evaluate( 'data/train_sub.txt', 'data/test.txt' )
# evaluate( 'data/train_noisy.txt', 'data/test.txt' )

# # 3.2:

# classifiers = cross_validation( 'data/train_full.txt', 10 )

# # 3.3:

# print("modal accuracy: ", modal_cross_validation( 'data/test.txt', classifiers ) )
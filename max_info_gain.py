#    See info_gain_notes.md if you need an explanation of the math.

import math # for log computation

def calculate_entropy(x, y, classes):
    '''
    This function calculates the entropy for a given dataset. 
    It's equivalent to formula (2) in the CW spec pg 6.
    Args:
        x: a 2D numpy matrice consisting of all the instances in the dataset
            and their corresponding attribute values
        y: a 1D numpy array consisting of the class labels of all instances
        classes: a 1D numpy array consisting of the 
    Returns:
        float: the entropy for the given dataset.
    '''
    total_number_of_instances = len(y)
    label_frequencies = dict.fromkeys(classes, 0)
    for label in y:
        label_frequencies[label] += 1
    
    entropy = 0.0
    for label, value in label_frequencies.items():
        if(value != 0):
            probability = value / total_number_of_instances
            if(probability > 0 and probability < 1):
                entropy -= probability * math.log(probability, 2)
    
    #TODO: add sanity check
    
    return entropy
    
    


# Main Function, follows pseudocode given in spec.
def induce_decision_tree(){
    ''' 
    This function computes the information gain for different subsets of
    the dataset passed in for different attribute and split points.
    Input: 
    '''


}


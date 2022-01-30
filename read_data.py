# for line in open("intro2ml_cw1_13/data/simple1.txt"):
#     print(line.strip())

from re import M
from attr import attr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_dataset(filepath):
    x = []
    y_labels = []
    for line in open(filepath):
        if line.strip() != "":
            row = line.strip().split(",")
            x.append(row[:-1])
            y_labels.append(row[-1])
    
    x = np.array(x).astype(np.int)
    [classes, y] = np.unique(y_labels, return_inverse=True)

    return (x, y, classes)

# displays the shape of x, y, and displays the number of each letter
def display_data_information(x,y,classes): 
    print("shape of x", end = " ")
    print(x.shape),
    print("shape of y", end = " ")
    print(y.shape),
    print("array of y labels", end = " ")
    print(classes)
    
    y_shape = []
    y_values = []
    temp = 0
    for letter in classes:
        y_shape.append(temp)
        temp = temp+1
    for number in y_shape:
        y_values.append(np.count_nonzero(y==number))
        print(np.count_nonzero(y==number), end=" "),
        print("datasets for", end=" "),
        print(classes[number])
    # print histogram

def understand_attributes(x):
    print("Minimum values for x", end=" ")
    print(x.min(axis=0))
    print("Maximum values for x", end=" ")
    print(x.max(axis=0))

## TAKES X, Y, AND ATTRIBUTE. returns a sorted array of x and y
def sort_list_by_attribute(x, y, attribute):
    sorted_x,sorted_y = x[np.argsort(x[:, attribute])], y[np.argsort(x[:,attribute])]
    return (sorted_x,sorted_y)

def only_certain_y(x, y, index):
    x_return = x[y==index]
    return x_return
# for line in open("intro2ml_cw1_13/data/simple1.txt"):
#     print(line.strip())

from re import M
from attr import attr, attrib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from torch import le

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

def get_probability_distribution(x,y,classes,letter,attribute):
    data = only_certain_y(x,y,letter)[:,attribute]
    plt.figure()
    plt.hist(data)
    plt.xlabel(f"Letter {classes[letter]}, attribute {attribute} value")
    plt.ylabel(f"Frequency distribution of attribute {attribute}")
    plt.show()

def get_probability_distribution_all_letters(x,y,classes,attribute):
    data = []
    for i in range(classes.size):
        data.append(only_certain_y(x,y,i)[:,attribute])
    
    plt.figure()
    for i in range(classes.size):
        plt.hist(data[i], alpha=0.5, label=classes[i])
    # plt.hist(data)
    plt.xlabel(f"Attribute {attribute} value")
    plt.ylabel(f"Frequency of attribute {attribute}")
    plt.legend(loc='upper right')
    plt.show()

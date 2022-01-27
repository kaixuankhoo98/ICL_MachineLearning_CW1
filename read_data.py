# for line in open("intro2ml_cw1_13/data/simple1.txt"):
#     print(line.strip())

from re import M
import numpy as np

def read_dataset(filepath):
    x = []
    y_labels = []
    for line in open(filepath):
        if line.strip() != "":
            row = line.strip().split(",")
            x.append(row[:-1])
            y_labels.append(row[-1])
    
    x = np.array(x)
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
    temp = 0
    for letter in classes:
        y_shape.append(temp)
        temp = temp+1
    for number in y_shape:
        print(np.count_nonzero(y==number), end=" "),
        print("datasets for", end=" "),
        print(classes[number])
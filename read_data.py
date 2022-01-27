# for line in open("intro2ml_cw1_13/data/simple1.txt"):
#     print(line.strip())

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

(x, y, classes) = read_dataset("intro2ml_cw1_13/data/simple2.txt")
print(x.shape)
print(y.shape)
print(classes)
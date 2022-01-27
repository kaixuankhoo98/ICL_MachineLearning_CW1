from calendar import c
import read_data as rd
import numpy as np
import matplotlib.pyplot as plt

(x, y, classes) = rd.read_dataset("intro2ml_cw1_13/data/train_full.txt")


rd.display_data_information(x,y,classes)

# y_shape = []
# temp = 0
# for letter in classes:
#     y_shape.append(temp)
#     temp = temp+1
# for number in y_shape:
#     print(np.count_nonzero(y==number), end=" "),
#     print("datasets for", end=" "),
#     print(classes[number])

# class_names = []
# for letter in classes:
#     class_names.append(letter)
# print(class_names)
# plt.figure()
# plt.scatter(x[:,0], x[:,1], c=y, cmap=plt.cm.Set1, edgecolor='k')
# plt.xlabel(class_names[0])
# plt.ylabel(class_names[1])
# plt.show()
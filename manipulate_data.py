from calendar import c
import read_data as rd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x, y, classes) = rd.read_dataset("intro2ml_cw1_13/data/toy.txt")

print("For train_sub.txt")
rd.display_data_information(x,y,classes)
new_list_x, new_list_y = rd.sort_list(x,y,0)
print(new_list_x, new_list_y)
# rd.understand_attributes(x)

# class_names = []
# for letter in classes:
#     class_names.append(letter)
# print(class_names)
# plt.figure()
# plt.scatter(x[:,0], x[:,1], c=y, cmap=plt.cm.Set1, edgecolor='k')
# # plt.xlabel(class_names[1])
# # plt.ylabel(class_names[2])
# plt.show()
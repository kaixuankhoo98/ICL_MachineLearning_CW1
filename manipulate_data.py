from calendar import c
import read_data as rd

# Don't need these in this file I think.
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

file_name = "data/toy.txt"

(x, y, classes) = rd.read_dataset(file_name)
# arg was "intro2ml_cw1_13/data/train_full.txt" just now.
# was getting an error-- fileNotFoundError.

print("For toy.txt")  # this was for "train_sub.txt". prob just a typo?
rd.display_data_information(x, y, classes)

new_list_x, new_list_y = rd.sort_list_by_attribute(x, y, 0)
# print(new_list_x, new_list_y)

new_x_onlyA = rd.only_certain_y(x, y, 0)
new_x_onlyC = rd.only_certain_y(x, y, 1)
print(new_x_onlyA)
print(new_x_onlyC)

rd.get_probability_distribution(x, y, 0, 0)

# class_names = []
# for letter in classes:
#     class_names.append(letter)
# print(class_names)
# plt.figure()
# plt.scatter(x[:,0], x[:,1], c=y, cmap=plt.cm.Set1, edgecolor='k')
# # plt.xlabel(class_names[1])
# # plt.ylabel(class_names[2])
# plt.show()

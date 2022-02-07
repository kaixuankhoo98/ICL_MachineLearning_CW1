from calendar import c
import read_data as rd
import numpy as np
import max_info_gain
from classification import DecisionTreeClassifier

# from max_info_gain import induce_decision_tree

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

# print("Trying info_gain function")
# max_info_gain.calculate_best_info_gain(x, y, classes)
# print("Now, trying induce decision tree")
# max_info_gain.induce_decision_tree(x, y, classes)

y_letters = []
for i in y:
    y_letters.append(classes[i])
y_letters = np.array(y_letters)
print("Trying fit method")
classifier = DecisionTreeClassifier()
classifier.fit(x, y_letters)
print("Trying predict method")
x_test = np.array([[5,7,1],[4,6,2],[4,6,3],[1,6,3],[0,5,5],[1,3,1],[2,1,2],[5,2,6],[1,5,0],[2,4,2]])
x_scrambled_test = np.array([[5,7,1],[1,3,1],[4,6,2],[2,1,2],[4,6,3],[5,2,6],[1,6,3],[1,5,0],[0,5,5],[2,4,2]])
print(classifier.predict(x_test))
print(classifier.predict(x_scrambled_test))
# new_list_x, new_list_y = rd.sort_list_by_attribute(x, y, 0)
# # print(new_list_x, new_list_y)

# new_x_onlyA = rd.only_certain_y(x, y, 0)
# new_x_onlyC = rd.only_certain_y(x, y, 1)
# print(new_x_onlyA)
# print(new_x_onlyC)

# rd.get_probability_distribution(x, y, 0, 0)

# print("\n")
# print(len(y))

# class_names = []
# for letter in classes:
#     class_names.append(letter)
# print(class_names)
# plt.figure()
# plt.scatter(x[:,0], x[:,1], c=y, cmap=plt.cm.Set1, edgecolor='k')
# # plt.xlabel(class_names[1])
# # plt.ylabel(class_names[2])
# plt.show()

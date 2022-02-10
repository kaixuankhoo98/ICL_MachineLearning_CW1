from calendar import c
import read_data as rd
import numpy as np
import max_info_gain
from classification import DecisionTreeClassifier
from evaluation import confusion_matrix, precision
from evaluation import accuracy
from evaluation import recall, precision, f1_score
import evaluation as ev
import cross_validation as cv

file_name = "data/train_sub.txt"

(x, y, classes) = rd.read_dataset(file_name)

print("For train_sub.txt") 
rd.display_data_information(x, y, classes)

# ===== ------- TESTING FIT AND PREDICT METHODS OF CLASSIFIER V1 ---- ======

y_letters = []
for i in y:
    y_letters.append(classes[i])
y_letters = np.array(y_letters)
print("Trying fit method")
classifier = DecisionTreeClassifier()
classifier.fit(x, y_letters)
print("Trying predict method on train_full dataset")
(x_full, y_full, classes_full) = rd.read_dataset("data/train_full.txt")
ypred=classifier.predict(x_full)
print(ypred)
# ===== ------- TESTING FIT AND PREDICT METHODS OF CLASSIFIER V1 ---- ======

# x_test = np.array([[5,7,1],[4,6,2],[4,6,3],[1,6,3],[0,5,5],[1,3,1],[2,1,2],[5,2,6],[1,5,0],[2,4,2]])
# x_scrambled_test = np.array([[5,7,1],[1,3,1],[4,6,2],[2,1,2],[4,6,3],[5,2,6],[1,6,3],[1,5,0],[0,5,5],[2,4,2]])
# print("Trying to print and traverse the tree")
# classifier.traverse_and_print_tree()



# evaluation.evaluate( "data/train_full.txt", "data/test.txt" )
# print('\n', '\n')
# evaluation.evaluate( "data/train_sub.txt", "data/test.txt" )
# print('\n', '\n')
# evaluation.evaluate( "data/train_noisy.txt", "data/test.txt" )



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
# print("Trying cross validation method")
# # TODO: resolve infinite loop issue in cross validation method
# ev.cross_validation("data/train_full.txt", 10)
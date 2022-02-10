# split instances into three bins
        bin_1_filter = x[:, attribute] < 5
        bin_2_filter = x[:, attribute] >= 5 and x[:, attribute] <= 10
        bin_3_filter = x[:, attribute] >= 10
        x_bin_1 = x[:, attribute][bin_1_filter]
        y_bin_1 = y[bin_1_filter]
        x_bin_2 = x[:, attribute][bin_2_filter]
        y_bin_2 = y[bin_2_filter]
        x_bin_3 = x[:, attribute][bin_3_filter]
        y_bin_3 = y[bin_3_filter]
        # calculate the entropy for each
        # DELIBERATELY INCOMPLETE: I don't think we should use this method.

        We will have 3 bins: when attribute value x is:
        . less than 5
        . between 5 and 10 (inclusive)
        . more than 10
     # TODO: consider alternative way to split: this feels like it'll overfit.
        e.g. it wouldn't really work for datasets that don't fall within those ranges.
        Another way would be to work with `max` and `min`.

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
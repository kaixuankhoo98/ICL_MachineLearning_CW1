import numpy as np
class Node(object):
    depth = 0
    left_child = None
    right_child = None
    last_node = False

    def __init__(self, attribute_index, attribute_value, x, y, classes):
        self.attribute_index = attribute_index
        self.attribute_value = attribute_value
        self.x = x
        self.y = y
        self.classes = classes

    def add_child(self, node):
        if self.left_child == None:
            self.left_child = node
            self.left_child.depth = self.depth + 1
        else:
            self.right_child = node
            self.right_child.depth = self.depth + 1

    # this function was taken from:
    # https://www.tutorialspoint.com/python_data_structure/python_tree_traversal_algorithms.htm
    # the section on 'PreOrder Traversal'
    def print_tree(self):
        if self.left_child:
            self.left_child.print_tree()
        print("Node: ")
        print(self.x)
        print(self.y)
        print(self.attribute_index)
        print(self.attribute_value)
        print("\n")
        if self.right_child:
            self.right_child.print_tree()

    # helper function for prune_tree-- checks if a given node's children are leaf nodes
    def children_are_leaves(self):
        if self.left_child.last_node and self.right_child.last_node:
            return True
        else:
            return False

    # helper function for prune_tree-- makes a node with children nodes that are leaf a leaf node
    # itself using the majority label from the children nodes.
    def make_node_leaf_node(self):
        # find majority label from children.
        self.x = np.concatenate(self.left_child.x, self.right_child.x)
        self.y = np.concatenate(self.left_child.y, self.right_child.y)
        #make into a leaf
        self.attribute_index = None
        self.attribute_value = None
        self.last_node = True
        self.left_child = None
        self.right_child = None

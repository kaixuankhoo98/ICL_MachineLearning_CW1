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
        if ( self.left_child == None ):
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
        print( self.attribute_index)
        print( self.attribute_value)
        print("\n")
        if self.right_child:
            self.right_child.print_tree()

    # helper function for prune_tree-- checks if a given node's children are leaf nodes
    def children_are_leaves(self, node_with_children):
        if(node_with_children.left_child.last_node and node_with_children.right_child.last_node):
            return True
        else:
            return False  

    # helper function for prune_tree-- makes a node with children nodes that are leaf a leaf node
    # itself using the majority label from the children nodes.
    def make_node_leaf_node(self, node_with_leaf_children):
        

    def prune_tree(self):
        # BASE CASE 1: if self has no children, then return

        if(self.left_child == None and self.right_child == None):
            return self
        if(self.children_are_leaves(self)):
            return self.make_node_leaf_node(self)
        # if self has children but not leaf children, then call the function again but on the children.
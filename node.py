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
        print("node: ")
        print( self.attribute_index)
        print( self.attribute_value)
        if self.right_child:
            self.right_child.print_tree()
    
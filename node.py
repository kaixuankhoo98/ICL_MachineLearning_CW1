class Node(object):


    left_child = None
    left_child_label = None
    right_child = None 
    right_child_label = None

    def __init__(self, attribute_index, attribute_value, x, y, classes):
        self.attribute_index = attribute_index
        self.attribute_value = attribute_value
        self.x = x
        self.y = y 
        self.classes = classes  
                   
    def add_child(self, node):
        if ( self.left_child == None ):
            self.left_child = node
        else:
            self.right_child = node
    
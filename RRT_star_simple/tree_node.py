import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

class TreeNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_node):
        self.children.append(child_node)

    def remove_child(self, child_node):
        self.children.remove(child_node)

def plot_tree(root):   
    if root is None:
        return
    plt.plot(root.state[0], root.state[1], 'ro')  # Plot the node
    for child in root.children:
        #plt.plot([root.state[0], child.state[0]], [root.state[1], child.state[1]], 'b-')  # Draw edge to child
        plt.arrow(root.state[0], root.state[1], child.state[0] - root.state[0], child.state[1] - root.state[1], shape='full', lw=0.1, length_includes_head=True, head_width=1, color='blue')
        plot_tree(child)  # Recursively plot children

def print_tree(node, level=0):
    if node is None:
        return
    print(" " * (level * 2) + str(node.state))
    for child in node.children:
        print_tree(child, level + 1)

def plot_path(node):
    if node is None:
        return
    if node.parent is None:
        return
    plot_path(node.parent)
    plt.arrow(node.parent.state[0], node.parent.state[1], 
             node.state[0] - node.parent.state[0], 
             node.state[1] - node.parent.state[1], 
             shape='full', lw=0.5, length_includes_head=True, head_width=2, color='green')
    
def get_path(node):
    path = []
    while node is not None:
        path.append(node.state)
        node = node.parent
    return path[::-1]  # Reverse the path to start from the root
if __name__ == "__main__":
    root = TreeNode(np.array([0, 0, 0]))
    child1 = TreeNode(np.array([1, 1, 0]))
    child2 = TreeNode(np.array([1, -1, 0]))
    root.add_child(child1)
    root.add_child(child2)
    target_state = np.array([0.1, 0, 0])

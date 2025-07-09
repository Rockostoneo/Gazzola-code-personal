import numpy as np
import numpy.linalg as la
from tree_node import TreeNode, plot_tree, print_tree
from rrstar_functions import sample, steer, nearest, near, cost
from celluloid import Camera
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from RRTstar import extend_RRT, extend_RRT_star


state_0 = np.array([0,0]) #initial x,y positions, and orientation theta
goal_coords = np.array([20,0]) # final x,y positions, and orientation doesnt matter but we set it to 0
x_bounds = [-100,100]
y_bounds = [-100,100]

nodes = 300
root = TreeNode(state_0)  # Initialize the tree with the root node
max_distance = 10
goal_sample_rate = 0.00 



root = TreeNode(np.array([0, 0, 0]))
child1 = TreeNode(np.array([1, 1, 0]))
child2 = TreeNode(np.array([1, -1, 0]))
child1.parent = root
child1.parent.add_child(child1)
#child1.parent = root
#child1.parent.remove_child(child1)
print_tree(root)
print(cost(child1))
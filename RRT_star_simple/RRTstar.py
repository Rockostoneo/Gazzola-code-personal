import numpy as np
import numpy.linalg as la
from tree_node import TreeNode, plot_tree, print_tree, plot_path, get_path
from rrstar_functions import sample, steer, nearest, near, cost, intersect
from celluloid import Camera
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

def extend_RRT(root, rand_state, max_distance):
    nearest_node = nearest(root, rand_state) 
    new_node = TreeNode(steer(nearest_node.state, rand_state, max_distance))
    if not intersect(nearest_node.state, new_node.state, lines):
        nearest_node.add_child(new_node)
    return

def extend_RRT_star(root, rand_state, max_distance, lines, goal_coords):
    nearest_node = nearest(root, rand_state) 
    new_node = TreeNode(steer(nearest_node.state, rand_state, max_distance),parent=nearest_node)
    if not intersect(nearest_node.state, new_node.state, lines):
        min_node = nearest_node
        near_nodes = near(root, new_node.state, max_distance)
        for near_node in near_nodes:
            if not intersect(near_node.state, new_node.state, lines):
                near_cost = cost(near_node) + la.norm(new_node.state - near_node.state)
                if near_cost < cost(new_node):
                    new_node.parent = near_node
                    min_node = near_node
        new_node.parent.add_child(new_node)        
        for near_node in near_nodes:
            if near_node != min_node:
                if not intersect(near_node.state, new_node.state, lines):
                    if (cost(new_node) + la.norm(new_node.state-near_node.state)) < cost(near_node):
                        near_node.parent.remove_child(near_node)
                        near_node.parent = new_node
                        new_node.add_child(near_node)
        if np.array_equal(new_node.state,goal_coords):
            return new_node 

if __name__ == "__main__":
    state_0 = np.array([0,0]) #initial x,y positions
    goal_coords = np.array([80,80]) # final x,y positions
    x_bounds = [-30,100]
    y_bounds = [-30,100]
    lines = [(0,80,40,20),
             (90,0,40,60)]
    lines = []
    nodes = 600
    root = TreeNode(state_0)  # Initialize the tree with the root node
    max_distance = 15
    goal_sample_rate = 0.15  # probability of sampling the goal state
    goal_node = None
    i = 0
    k = 0   
    random.seed(33)

    fig = plt.figure()
    camera = Camera(fig)
    plt.xlim(x_bounds)
    plt.ylim(y_bounds)
    plt.gca().set_aspect('equal', adjustable='box')
    for i in tqdm(range(nodes)):
        rand_state = sample((x_bounds, y_bounds), goal_coords, goal_sample_rate, goal_node)  
        temp = extend_RRT_star(root, rand_state, max_distance, lines, goal_coords) #right now there are duplicate goal nodes
        if temp is not None:
            goal_node = temp
        #plot_tree(root)
        #plt.plot(state_0[0], state_0[1], 'bo')
        #plt.plot(goal_coords[0], goal_coords[1], 'bo') 
        #for line in lines:
        #    plt.plot(line[0::2], line[1::2], 'k-')
        #if goal_node is not None:
        #    plot_path(goal_node)
        #camera.snap()

    print(get_path(goal_node))  # Print the path from root to goal node
    #print_tree(root)  # Print the tree structure
    print("Creating animation...")
    #animation = camera.animate() #TODO speed this up
    #animation.save('animation.mp4')
    plt.clf()
    plot_tree(root)
    plt.plot(state_0[0], state_0[1], 'bo')
    plt.plot(goal_coords[0], goal_coords[1], 'bo') 
    for line in lines:
        plt.plot(line[0::2], line[1::2], 'k-')
    if goal_node is not None:
            plot_path(goal_node)
    plt.show()
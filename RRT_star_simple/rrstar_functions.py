import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import random

def sample(bounds,goal_state,goal_sample_rate, goal_node): 
    if (random.random() < goal_sample_rate) and (goal_node is None):
        return goal_state
    x_bounds, y_bounds = bounds
    x = random.uniform(x_bounds[0], x_bounds[1])
    y = random.uniform(y_bounds[0], y_bounds[1])
    return np.array([x, y])

def steer(near_state, rand_state, max_distance):
    if la.norm(rand_state - near_state) <= max_distance:
        return rand_state
    else:
        return near_state + (rand_state - near_state) * (max_distance / la.norm(rand_state - near_state))
        

def nearest(node, target_state):
    if node is None:
        return
    best_node = node
    for child in node.children:
        best_child = nearest(child, target_state)
        child_state = best_child.state
        if la.norm(child_state - target_state) <la.norm(best_node.state - target_state):
            best_node = best_child
    return best_node

def near(node, target_state, radius):
    if node is None:
        return []
    near_nodes = []
    if la.norm(node.state - target_state) <= radius:
        near_nodes.append(node)
    for child in node.children:
        near_nodes.extend(near(child, target_state, radius))
    return near_nodes

def cost(node):
    if node is None:
        return 0
    return la.norm(node.state-node.parent.state) + cost(node.parent) if node.parent else 0

def intersect(a,b, lines):  #TODO rewrite to not use division
    for line in lines:
        uA = ((line[2]-line[0])*(a[1]-line[1]) - (line[3]-line[1])*(a[0]-line[0])) / ((line[3]-line[1])*(b[0]-a[0]) - (line[2]-line[0])*(b[1]-a[1]))
        uB = ((b[0]-a[0])*(a[1]-line[1]) - (b[1]-a[1])*(a[0]-line[0])) / ((line[3]-line[1])*(b[0]-a[0]) - (line[2]-line[0])*(b[1]-a[1]))
        if 0 <= uA <= 1 and 0 <= uB <= 1:
            return True
    return False

if __name__ == "__main__":
    lines = [(50,10,10,50),
             (20,30,30,20)]
    print(intersect((13.9424994,50.60526442),(7.47636225, 37.07052872),lines))  


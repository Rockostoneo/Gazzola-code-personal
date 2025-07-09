import numpy as np
import random
from tqdm import tqdm
import control as ct
#visualization
import matplotlib.pyplot as plt

#Custom modules
from rrstar_functions import sample, LQR_steer, LQR_nearest, LQR_near, choose_parent, collision_free, rewire, customlinearize


if __name__ == "__main__":
    state_0 = np.array([0,0,0]) #initial x, y position, and orientation theta
    goal_coords = np.array([30,30,0]) # final x, y position, and orientation doesnt matter 
    goal_coords[2] = np.arctan2(goal_coords[1]-state_0[1], goal_coords[0]-state_0[0]) % (2*np.pi) # Set orientation towards the goal #TODO fix orientation
    x_bounds = [-30,100]
    y_bounds = [-30,100]
    lines = [(0,80,40,20),
             (90,0,40,60)]

    n = 50 # number of nodes to sample
    G_nodes = np.inf * np.ones((n+1, 4)) # each row is a node with [x, y, theta, cost]
    G_edges = np.inf * np.ones((n, 4)) #each row contains endnodes and Action
    nun = 0 # number of nodes so far (0 indexed)

    gamma = 200000  # tuning parameter for the near function
    max_distance = gamma * (np.log(n)/n)**(1/3)  
    goal_sample_rate = 0.9  # probability of sampling the goal state
    h = 1  # time step for the system dynamics
    G_nodes[0, :3] = state_0  # Set the initial state
    G_nodes[0, 3] = 0  # Set the cost of the initial state to 0
    # LQR parameters 
    Q = np.diag([1, 1, 0.05])  # State cost matrix
    R = np.diag([2, 1])  # Control cost matrix
    random.seed(8)
    r = 3
    u_bounds = 1  # Steering input bounds
    V_max = 200
    for i in tqdm(range(n)):
        Steer_Action = [np.inf,0]
        while (np.abs(Steer_Action[0]) > u_bounds) or (Steer_Action[1] < 0) or (Steer_Action[1] > V_max): #ensure action is within control bounds 
            rand_state = sample((x_bounds, y_bounds), goal_coords, goal_sample_rate)  #sample goal with a specific angle or not?
            A,B = customlinearize(rand_state, r)  
            K_rand, S_rand, E_rand = ct.lqr(A, B, Q, R)
            id_nearest, x_nearest = LQR_nearest(rand_state, S_rand, nun, G_nodes)  # Find the nearest node in the tree
            Steer_Action = -K_rand @ (x_nearest - rand_state)
        x_new = LQR_steer(x_nearest, rand_state, K_rand, r, h)  
        A,B = customlinearize(x_new, r)  # Linearize the system at the new state
        K_new, S_new, E_new = ct.lqr(A, B, Q, R)
        X_near_ids = LQR_near(x_new, S_new, max_distance, nun, G_nodes) 
        id_parent, x_parent, min_cost = choose_parent(X_near_ids, x_new, S_new, G_nodes)
        to_parent_cost = G_nodes[id_parent,3]
        to_new_cost = to_parent_cost + (x_parent - x_new).T @ S_new @ (x_parent - x_new)  # Cost to the new node
        nun = nun + 1
        G_nodes[nun, :3] = x_new
        G_nodes[nun, 3] = to_new_cost
        action = -K_new @ (x_parent-x_new)  # Calculate the control action to reach the new node
        G_edges[nun-1, 0] = id_parent
        G_edges[nun-1, 1] = nun
        G_edges[nun-1, 2] = action[0]
        G_edges[nun-1, 3] = action[1]
        rewire(X_near_ids, x_new, nun,G_nodes, G_edges, Q, R, r, u_bounds, V_max)  # Rewire the tree if necessary
        #TODO implement pruning?, implement collision checking

    fig = plt.figure()
    plt.quiver(G_nodes[:nun+1, 0], G_nodes[:nun+1, 1], np.cos(G_nodes[:nun+1, 2]), np.sin(G_nodes[:nun+1, 2]), angles = "xy", pivot='middle', color='b', scale=50)
    plt.plot(goal_coords[0], goal_coords[1], 'ro')  # Plot the goal state
    plt.plot(G_nodes[0, 0], G_nodes[0, 1], 'go')  # Plot the start state

    for i in range(n):
        plt.plot([G_nodes[int(G_edges[i, 0]), 0], G_nodes[int(G_edges[i, 1]), 0]], 
                 [G_nodes[int(G_edges[i, 0]), 1], G_nodes[int(G_edges[i, 1]), 1]], 'k-')
    plt.show()


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import random
import control as ct

def customlinearize(x, r, u = [0.0,1.0]): #linearization hard coded for the snake system to speed up code
    A = np.array([[0, 0, -u[1] * np.sin(x[2])],
                  [0, 0, u[1] * np.cos(x[2])],
                  [0, 0, 0]])
    B = np.array([[0, np.cos(x[2])],
                  [0, np.sin(x[2])],
                  [u[1]/r, 0]]) # type: ignore
    return A, B

def sample(bounds, goal_state, goal_sample_rate): 
    if random.random() < goal_sample_rate:
        return goal_state
    x_bounds, y_bounds = bounds
    x = random.uniform(x_bounds[0], x_bounds[1])
    y = random.uniform(y_bounds[0], y_bounds[1])
    theta = random.uniform(0, 2* np.pi) # is 0-2pi or -pi to pi better?
    return np.array([x, y, theta])

def intersect(a,b, lines):  #TODO rewrite to not use division
    for line in lines:
        uA = ((line[2]-line[0])*(a[1]-line[1]) - (line[3]-line[1])*(a[0]-line[0])) / ((line[3]-line[1])*(b[0]-a[0]) - (line[2]-line[0])*(b[1]-a[1]))
        uB = ((b[0]-a[0])*(a[1]-line[1]) - (b[1]-a[1])*(a[0]-line[0])) / ((line[3]-line[1])*(b[0]-a[0]) - (line[2]-line[0])*(b[1]-a[1]))
        if 0 <= uA <= 1 and 0 <= uB <= 1:
            return True
    return False

def LQR_nearest(x_rand, S_rand, nun, G_nodes):
    nearest_cost = np.inf
    x_nearest = G_nodes[0, :3]
    nearest_node = 0
    for i in range(nun+1):
        x = G_nodes[i, :3]
        cost = (x - x_rand).T @ S_rand @ (x - x_rand)
        if cost < nearest_cost:
            nearest_cost = cost
            nearest_node = i
            x_nearest = x
    return nearest_node, x_nearest

def LQR_steer(x_nearest, x_rand, K_rand,r,h):
    u = -K_rand @ (x_nearest - x_rand)
    x_dot = np.array([np.cos(x_nearest[2]) * u[1],
                      np.sin(x_nearest[2]) * u[1],  
                      u[0] * u[1] / r])
    return x_nearest + x_dot * h  

def LQR_near(x_new, S_new, max_cost, nun,G_nodes):
    X_near_ids = []
    for i in range(nun+1):
        x = G_nodes[i, :3]
        cost = (x - x_new).T @ S_new @ (x - x_new)
        if cost < max_cost:
            X_near_ids.append((i))
    return X_near_ids

def collision_free(a, b, lines): #TODO: implement collision checking
    return True 

def rewire(X_near_ids, x_new, id_new_node,G_nodes,G_edges, Q, R, r, u_bounds, V_max):  #TODO: steering and velocity limits for rewire
    to_newCost = G_nodes[id_new_node, 3]
    for i in X_near_ids:
        x = G_nodes[i, :3]
        to_canCost = G_nodes[i, 3] 
        A, B = customlinearize(x, r)
        K, S, E = ct.lqr(A, B, Q, R)
        ntocCost = (x_new - x).T @ S @ (x_new - x) 
        Steer_Action = -K @ (x_new - x)
        if to_newCost + ntocCost < to_canCost and (np.abs(Steer_Action[0]) < u_bounds) and (0 < Steer_Action[1] < V_max):
            G_nodes[i, 3] = to_newCost + ntocCost
            edge_index = np.where(G_edges[:, 1] == i)  #TODO might not be right
            G_edges[edge_index, 0] = id_new_node
            G_edges[edge_index, 1] = i
            G_edges[edge_index, 2] = (-K @ (x_new - x))[0] 
            G_edges[edge_index, 3] = (-K @ (x_new - x))[0]
            
def choose_parent(X_near_ids, x_new, S_new,G_nodes):
    min_cost = np.inf
    for i in X_near_ids:
        x_near = G_nodes[i, :3]
        compare_cost = (x_near - x_new).T @ S_new @ (x_near - x_new) + G_nodes[i, 3]
        if compare_cost < min_cost:
            min_cost = compare_cost
            id_parent = i
            x_parent = x_near
    return id_parent, x_parent, min_cost

if __name__ == "__main__":
    lines = [(50,10,10,50),
             (20,30,30,20)]
    print(intersect((13.9424994,50.60526442),(7.47636225, 37.07052872),lines))  


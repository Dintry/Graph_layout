from __future__ import division
import networkx as nx
import numpy as np
import copy
import matplotlib.pyplot as plt


def initial_coords(G, radius=0.5):
    coords = {}
    N = len(G.nodes())
    for i in xrange(N):
        low = - radius / 2 ** 0.5
        high = -low
        coords[i] = np.random.uniform(low, high, 2)
    return coords


def farthest_node(G, centers):
    path_length = nx.get_node_attributes(G,'path_length')
    all_nodes = G.nodes()
    dist_from_set = 0
    farthest_node = 0
    for node in all_nodes:
        new_dist_from_set = 1000
        for c_node in centers:
            if new_dist_from_set > path_length[node][c_node]:
                new_dist_from_set = path_length[node][c_node]
        if dist_from_set < new_dist_from_set:
            dist_from_set = new_dist_from_set
            farthest_node = node
    return farthest_node, dist_from_set


def k_centers(G, k):
    centers = []
    centers.append(G.nodes()[0])
    radius = 0
    while len(centers) < k:
        new_center, dist_from_set = farthest_node(G, centers)
        centers.append(new_center)
        if radius < dist_from_set:
            radius = dist_from_set
    return centers, radius


def calculate_delta_value(G, node, rad):
    coords = nx.get_node_attributes(G, 'coords')
    path_length = nx.get_node_attributes(G, 'path_length')[node]
    spring_strength = nx.get_node_attributes(G, 'spring_strength')[node]
    spring_length = nx.get_node_attributes(G, 'spring_length')[node]
    E_x = 0
    E_y = 0
    N = len(coords)
    for j in G.nodes():
        if (path_length[j] < rad) & (node != j):
            distance = np.linalg.norm(coords[node] - coords[j])
            E_x += spring_strength[j] * (coords[node][0] - coords[j][0]) * (1 - spring_length[j] / distance)
            E_y += spring_strength[j] * (coords[node][1] - coords[j][1]) * (1 - spring_length[j] / distance)
    delta = (E_x ** 2 + E_y ** 2) ** 0.5
    return(delta, E_x, E_y)


def calculate_displacement(G, node, rad, E_x, E_y):
    coords = nx.get_node_attributes(G, 'coords')
    path_length = nx.get_node_attributes(G, 'path_length')[node]
    spring_strength = nx.get_node_attributes(G, 'spring_strength')[node]
    spring_length = nx.get_node_attributes(G, 'spring_length')[node]
    E_xx = 0
    E_xy = 0
    E_yy = 0
    N = len(coords)
    for j in G.nodes():
        if (path_length[j] < rad) & (node != j):
            distance = np.linalg.norm(coords[node] - coords[j])
            E_xx += spring_strength[j] * (1 - spring_length[j] / distance + spring_length[j] * (coords[node][0] - coords[j][0]) ** 2 / distance ** 3)
            E_yy += spring_strength[j] * (1 - spring_length[j] / distance + spring_length[j] * (coords[node][1] - coords[j][1]) ** 2 / distance ** 3)
            E_xy += spring_strength[j] * (coords[node][0] - coords[j][0]) * (coords[node][1] - coords[j][1]) * spring_length[j] / distance ** 3
    a = np.array([[E_xx, E_xy], [E_xy, E_yy]])
    b = np.array([-E_x, -E_y])
    displacement = np.linalg.solve(a, b)
    return(displacement)


def local_layout(G, rad, iterations):
    coords = nx.get_node_attributes(G, 'coord')
    nodes = G.nodes()
    delta = {}
    E_x = {}
    E_y = {}
    N = len(coords)
    for i in xrange(iterations * N):
        max_elem = nodes[0]
        max_delta = 0
        for node in nodes:
            delta[node], E_x[node], E_y[node] = calculate_delta_value(G, node, rad)
            if max_delta < delta[node]:
                max_elem = node
                max_delta = delta[node]
        disp = calculate_displacement(G, max_elem, rad, E_x[max_elem], E_y[max_elem])
        coords[max_elem] += disp
    return(coords)


def closest_center(G, node, centers):
    path_length = nx.get_node_attributes(G, 'path_length')
    closest_center = centers[0]
    for center in centers:
        if path_length[node][closest_center] > path_length[node][center]:
            closest_center = center
    return closest_center


def harel_koren(G, rad=7, iterations=4, ratio=3, min_size=10, length=1):
    k = min_size
    path_length = nx.shortest_path_length(G)
    spring_length = copy.deepcopy(path_length)
    spring_strength = copy.deepcopy(path_length)
    nodes = G.nodes()
    coords = initial_coords(G)
    for i in nodes:
        for j in nodes:
            if j != i:
                spring_length[i][j] = path_length[i][j] * length
                spring_strength[i][j] = 1 / (path_length[i][j] ** 2)
    while k <= len(G.nodes()):
        centers, radius = k_centers(G, k)
        radius *= rad
        #local_G = nx.Graph(G.subgraph(centers))
        #local_coords = {}
        #local_path_length = {}
        #local_spring_strength = {}
        #local_spring_length = {}
        #for v in centers:
        #    local_coords[v] = coords[v]
        #    local_path_length[v] = {}
        #    local_spring_strength[v] = {}
        #    local_spring_length[v] = {}
        #    for u in centers:
        #        local_path_length[v][u] = path_length[v][u]
        #        local_spring_strength[v][u] = spring_strength[v][u]
        #        local_spring_length[v][u] = spring_length[v][u]
        #nx.set_node_attributes(local_G, 'spring_length', local_spring_length)
        #nx.set_node_attributes(local_G, 'spring_strength', local_spring_strength)
        #nx.set_node_attributes(local_G, 'path_length', local_path_length)
        #nx.set_node_attributes(local_G, 'coords', local_coords)
        local_coords = local_layout(local_G, rad, iterations)
        for node in nodes:
            coords[node] = local_coords[closest_center(G, node, centers)] + np.random.rand(2)
        k *= ratio
    return coords


N = 50
G = nx.barabasi_albert_graph(N, 2)
dist_i_j = nx.shortest_path_length(G)
nx.set_node_attributes(G, 'path_length', dist_i_j)

nx_pos = nx.spring_layout(G)
plt.figure(figsize=(10, 10))
nx.draw_networkx(G, nx_pos, with_labels=False)
plt.show()

new_coords = harel_koren(G)
plt.figure(figsize=(10, 10))
nx.draw_networkx(G, new_coords, with_labels=False)
plt.show()

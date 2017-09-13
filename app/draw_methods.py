from __future__ import division
import networkx as nx
import numpy as np
import copy


def initial_coords(G, radius):
    coords = {}
    N = len(G.nodes())
    for i in xrange(N):
        low = - radius / 2 ** 0.5
        high = -low
        coords[i] = np.random.uniform(low, high, 2)
    return coords


def calculate_delta_value(coords, k_i, l_i, i):
    E_x = 0
    E_y = 0
    N = len(coords)
    for j in range(N):
        if j != i:
            d_i_j = np.linalg.norm(coords[i] - coords[j])
            E_x += k_i[j] * (coords[i][0] - coords[j][0]) * (1 - l_i[j] / d_i_j)
            E_y += k_i[j] * (coords[i][1] - coords[j][1]) * (1 - l_i[j] / d_i_j)
    delta = (E_x ** 2 + E_y ** 2) ** 0.5
    return delta, E_x, E_y


def calculate_displacement(coords, k_i, l_i, i, E_x, E_y):
    E_xx = 0
    E_xy = 0
    E_yy = 0
    N = len(coords)
    for j in range(N):
        if j != i:
            d_i_j = np.linalg.norm(coords[i] - coords[j])
            E_xx += k_i[j] * (1 - l_i[j] / d_i_j + l_i[j] * (coords[i][0] - coords[j][0]) ** 2 / d_i_j ** 3)
            E_yy += k_i[j] * (1 - l_i[j] / d_i_j + l_i[j] * (coords[i][1] - coords[j][1]) ** 2 / d_i_j ** 3)
            E_xy += k_i[j] * (coords[i][0] - coords[j][0]) * (coords[i][1] - coords[j][1]) * l_i[j] / d_i_j ** 3
    a = np.array([[E_xx, E_xy], [E_xy, E_yy]])
    b = np.array([-E_x, -E_y])
    displacement = np.linalg.solve(a, b)
    return displacement


def kamada_kawai(G, k):
    eps = 0.1
    length = 1
    coords = initial_coords(G)
    paths = nx.shortest_path_length(G)
    diameter = nx.diameter(G)
    desirable_length = length / diameter
    ideal_length = copy.deepcopy(paths)
    spring_strength = copy.deepcopy(paths)
    nodes = G.nodes()
    for i in paths.keys():
        for j in paths[i].keys():
            if i != j:
                ideal_length[i][j] = paths[i][j] * desirable_length
                spring_strength[i][j] = k / (paths[i][j] ** 2)
    delta = np.ones(len(nodes))
    E_x = np.zeros(len(nodes))
    E_y = np.zeros(len(nodes))
    t = 1000
    while (max(delta) > eps) and (t > 0):
        for node in nodes:
            delta[node], E_x[node], E_y[node] = calculate_delta_value(coords, spring_strength[node], ideal_length[i],
                                                                      node)
        max_elem = np.argmax(delta)
        while delta[max_elem] > eps:
            disp = calculate_displacement(coords, spring_strength[max_elem], ideal_length[max_elem], max_elem,
                                          E_x[max_elem], E_y[max_elem])
            coords[max_elem] += disp
            delta[max_elem], E_x[max_elem], E_y[max_elem] = calculate_delta_value(coords, spring_strength[max_elem],
                                                                                  ideal_length[max_elem], max_elem)
        t -= 1
    return coords


def attractive_force(distance, k):
    return distance ** 2 / k


def repulsive_force(distance, k):
    return k ** 2 / distance


def cool(temperature):
    temperature *= 0.99
    return temperature


def calc_new_coords(coords, disp, radius, temperature):
    disp = (disp / np.linalg.norm(disp)) * min(np.linalg.norm(disp), temperature)
    new_coords = coords + disp
    if radius ** 2 < (new_coords[0] ** 2 + new_coords[1] ** 2):
        ''' 
        1) x + Ay + B = 0
           x**2 + y**2 + C = 0
        2) x = -Ay - B
           (Ay)**2 + 2ABy + B**2 + y**2 + C = 0
        3) x = -Ay - B
           (A**2 + 1)y + 2ABy + B**2 + C = 0
        4) x = -Ay - B   
           y = (-AB +- sqrt(-B**2-(A**2)C-C) / (A**2 + 1)
        '''
        B = (-disp[1] * coords[0] + disp[0] * coords[1]) / disp[1]
        A = -disp[0] / disp[1]
        C = -radius ** 2
        y = (-A * B + (-B ** 2 - C * A ** 2 - C) ** 0.5) / (A ** 2 + 1)
        x = -A * y - B
        if (x * disp[0] + y * disp[1]) < 0:
            y = (-A * B - (-B ** 2 - C * A ** 2 - C) ** 0.5) / (A ** 2 + 1)
            x = -A * y - B
        new_coords[0] = x
        new_coords[1] = y
    return new_coords


def fruchterman_reingold(G, eps=0.005, radius=0.5):
    temperature = 0.5 * radius
    N = len(G.nodes())
    area = 3.14 * radius ** 2
    k = (area / N) ** 0.5
    coords = initial_coords(G, radius)
    flag = 1
    while flag != 0:
        displacement = np.zeros((N, 2))
        for i in xrange(N):
            for j in xrange(N):
                if i != j:
                    delta = coords[i] - coords[j]
                    displacement[i] += (delta / np.linalg.norm(delta)) * repulsive_force(np.linalg.norm(delta), k)
        for edge in G.edges():
            delta = coords[edge[0]] - coords[edge[1]]
            displacement[edge[0]] -= (delta / np.linalg.norm(delta)) * attractive_force(np.linalg.norm(delta), k)
            displacement[edge[1]] += (delta / np.linalg.norm(delta)) * attractive_force(np.linalg.norm(delta), k)
        delta = 0
        old_coords = coords.copy()
        for i in xrange(1, N):
            coords[i] = calc_new_coords(coords[i], disp[i], radius, temperature)
            delta = max(delta, sum([(old_coords[i][0] - coords[i][0]) ** 2, (old_coords[i][1] - coords[i][1]) ** 2]))
        temperature = cool(temperature)
        if delta <= eps:
            flag = 0
    return coords

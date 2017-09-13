import networkx as nx
import numpy as np
import copy


def initial_layout(graph, radius=1):
    num = len(graph.nodes())
    coordinates = {}
    for i in xrange(num):
        low = - radius / 2 ** 0.5
        high = -low
        coordinates[i] = np.random.uniform(low, high, 2)
    return coordinates


def calculate_delta_value(coordinates, ideal_length, spring_strength, i):
    num = len(coordinates)
    distance = np.zeros(num)
    e_x = np.zeros(num)
    e_y = np.zeros(num)

    for j in xrange(num):
        if j != i:
            distance[j] = sum((coordinates[i] - coordinates[j]) ** 2) ** 0.5
            e_x[j] = spring_strength[j] * (coordinates[i][0] - coordinates[j][0]) * (1 - ideal_length[j] / distance[j])
            e_y[j] = spring_strength[j] * (coordinates[i][1] - coordinates[j][1]) * (1 - ideal_length[j] / distance[j])
    delta = (np.sum(e_x) ** 2 + np.sum(e_y) ** 2) ** 0.5

    return delta, e_x, e_y, distance


def calculate_delta_values(coordinates, ideal_length, spring_strength):
    num = len(coordinates)
    e_x = np.zeros((num, num))
    e_y = np.zeros((num, num))
    distance = np.zeros((num, num))

    for i in xrange(num):
        for j in xrange(i + 1, num):
            distance[i][j] = np.sum((coordinates[i] - coordinates[j]) ** 2) ** 0.5
            distance[j][i] = distance[i][j]
            e_x[i][j] = spring_strength[i][j] * (coordinates[i][0] - coordinates[j][0]) * (
                        1 - ideal_length[i][j] / distance[i][j])
            e_x[j][i] = -e_x[i][j]
            e_y[i][j] = spring_strength[i][j] * (coordinates[i][1] - coordinates[j][1]) * (
                        1 - ideal_length[i][j] / distance[i][j])
            e_y[j][i] = -e_y[i][j]

    delta = (np.sum(e_x, axis=1) ** 2 + np.sum(e_y, axis=1) ** 2) ** 0.5

    return delta, e_x, e_y, distance


def calculate_displacement(coordinates, ideal_length, spring_strength, i, e_x, e_y, distance):
    num = len(coordinates)
    e_x = np.sum(e_x)
    e_y = np.sum(e_y)
    e_xx = 0
    e_xy = 0
    e_yy = 0

    for j in xrange(num):
        if j != i:
            frac = ideal_length[j] / distance[j]
            e_xx += spring_strength[j] * (1 - frac * (1 - ((coordinates[i][0] - coordinates[j][0]) / distance[j]) ** 2))
            e_yy += spring_strength[j] * (1 - frac * (1 - ((coordinates[i][1] - coordinates[j][1]) / distance[j]) ** 2))
            e_xy += spring_strength[j] * (coordinates[i][0] - coordinates[j][0]) * (
                coordinates[i][1] - coordinates[j][1]) * ideal_length[j] / distance[j] ** 3

    ddx = (e_y * e_xy - e_x * e_yy) / (e_xx * e_yy - e_xy ** 2)
    ddy = (e_x * e_xy - e_y * e_xx) / (e_xx * e_yy - e_xy ** 2)

    displacement = np.array([ddx, ddy])

    return displacement


def kamada_kawai_alg(graph, options=(0.001, 1000)):
    eps, temp = options

    coordinates = nx.get_node_attributes(graph, 'coordinates')
    ideal_length = nx.get_node_attributes(graph, 'ideal_length')
    spring_strength = nx.get_node_attributes(graph, 'spring_strength')

    num = len(coordinates)

    delta, e_x, e_y, distance = calculate_delta_values(coordinates, ideal_length, spring_strength)
    while (max(delta) > eps) and (temp >= 0):
        max_elem = np.argmax(delta)
        delta_node = delta[max_elem]
        e_x_node = e_x[max_elem]
        e_y_node = e_y[max_elem]
        distance_node = distance[max_elem]
        ideal_length_node = ideal_length[max_elem]
        spring_strength_node = spring_strength[max_elem]

        jj = 0
        while (delta_node > eps) & (jj < num):
            jj += 1
            displacement = calculate_displacement(coordinates, ideal_length_node, spring_strength_node, max_elem,
                                                  e_x_node, e_y_node, distance_node)

            coordinates[max_elem] += displacement
            delta_node, e_x_node, e_y_node, distance_node = calculate_delta_value(coordinates, ideal_length_node,
                                                                                  spring_strength_node, max_elem)
        distance[max_elem], distance[:, max_elem] = distance_node, distance_node
        e_x[max_elem], e_x[:, max_elem] = e_x_node, -e_x_node
        e_y[max_elem], e_y[:, max_elem] = e_y_node, -e_y_node
        delta = (np.sum(e_x, axis=1) ** 2 + np.sum(e_y, axis=1) ** 2) ** 0.5
        temp -= 1
    return coordinates


def attractive_force(distance, spring_strength):
    return distance ** 2 / spring_strength


def repulsive_force(distance, spring_strength):
    return spring_strength ** 2 / distance


def cool(temperature):
    temperature *= 0.99
    return temperature


def calculate_new_coordinates(coordinates, displacement, radius, temperature):
    displacement = (displacement / np.linalg.norm(displacement)) * min(np.linalg.norm(displacement), temperature)
    new_coordinates = coordinates + displacement
    if radius ** 2 < (new_coordinates[0] ** 2 + new_coordinates[1] ** 2):
        ''' 
        1) x + ay + b = 0
           x**2 + y**2 + c = 0
        2) x = -ay - b
           (ay)**2 + 2aby + b**2 + y**2 + c = 0
        3) x = -ay - b
           (a**2 + 1)y + 2aby + b**2 + c = 0
        4) x = -ay - b   
           y = (-ab +- sqrt(-b**2-(a**2)c-c) / (a**2 + 1)
        '''
        b = (-displacement[1] * coordinates[0] + displacement[0] * coordinates[1]) / displacement[1]
        a = -displacement[0] / displacement[1]
        c = -radius ** 2
        y = (-a * b + (-b ** 2 - c * a ** 2 - c) ** 0.5) / (a ** 2 + 1)
        x = -a * y - b
        if (x * displacement[0] + y * displacement[1]) < 0:
            y = (-a * b - (-b ** 2 - c * a ** 2 - c) ** 0.5) / (a ** 2 + 1)
            x = -a * y - b
        new_coordinates[0] = x
        new_coordinates[1] = y
    return new_coordinates


def fruchterman_reingold(graph, eps=0.005, radius=0.5):
    temperature = 0.5 * radius
    num = len(graph.nodes())
    area = 3.14 * radius ** 2
    k = (area / num) ** 0.5
    coordinates = initial_layout(graph, radius)
    flag = 1
    while flag != 0:
        displacement = np.zeros((num, 2))
        for i in xrange(num):
            for j in xrange(num):
                if i != j:
                    delta = coordinates[i] - coordinates[j]
                    displacement[i] += (delta / np.linalg.norm(delta)) * repulsive_force(np.linalg.norm(delta), k)
        for edge in graph.edges():
            delta = coordinates[edge[0]] - coordinates[edge[1]]
            displacement[edge[0]] -= (delta / np.linalg.norm(delta)) * attractive_force(np.linalg.norm(delta), k)
            displacement[edge[1]] += (delta / np.linalg.norm(delta)) * attractive_force(np.linalg.norm(delta), k)
        delta = 0
        old_coordinates = coordinates.copy()
        for i in xrange(1, num):
            coordinates[i] = calculate_new_coordinates(coordinates[i], displacement[i], radius, temperature)
            delta = max(delta, sum(
                [(old_coordinates[i][0] - coordinates[i][0]) ** 2, (old_coordinates[i][1] - coordinates[i][1]) ** 2]))
        temperature = cool(temperature)
        if delta <= eps:
            flag = 0
    return coordinates


def kamada_kawai(graph, options=[0.1, 100.0, 0.1, 1000]):

    length, k, eps, initial_temp = options
    alg_options = (eps, initial_temp)

    num = len(graph.nodes())
    coordinates = initial_layout(graph)
    path_length = nx.shortest_path_length(graph)

    ideal_length = copy.deepcopy(path_length)
    spring_strength = copy.deepcopy(path_length)

    for i in xrange(num):
        for j in xrange(i + 1, num):
            ideal_length[i][j] *= length
            ideal_length[j][i] = ideal_length[i][j]
            spring_strength[i][j] = k / (spring_strength[i][j] ** 2)
            spring_strength[j][i] = spring_strength[i][j]

    nx.set_node_attributes(graph, 'coordinates', coordinates)
    nx.set_node_attributes(graph, 'ideal_length', ideal_length)
    nx.set_node_attributes(graph, 'spring_strength', spring_strength)

    new_coordinates = kamada_kawai_alg(graph, alg_options)

    return new_coordinates

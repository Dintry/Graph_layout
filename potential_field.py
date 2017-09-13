import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy


def create_graph(create_options=(20, 3), graph_type='barabasi_albert_graph'):
    create_options = ','.join(map(str, create_options))
    graph = eval('nx.' + graph_type + '(' + create_options + ')')

    nodes = {}
    i = 0
    for node in graph.nodes():
        if node not in nodes.keys():
            nodes[node] = i
            i += 1
    edges = []
    for edge in graph.edges():
        edges.append((nodes[edge[0]], nodes[edge[1]]))

    graph = nx.Graph()
    graph.add_nodes_from(nodes.values())
    graph.add_edges_from(edges)

    return graph


def initial_layout(graph):
    num = len(graph.nodes())
    coordinates = {}
    for i in xrange(num):
        coordinates[i] = np.random.rand(2)
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
            e_x[i][j] = spring_strength[i][j] * (coordinates[i][0] - coordinates[j][0]) * (1 - ideal_length[i][j] / distance[i][j])
            e_x[j][i] = -e_x[i][j]
            e_y[i][j] = spring_strength[i][j] * (coordinates[i][1] - coordinates[j][1]) * (1 - ideal_length[i][j] / distance[i][j])
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


def kamada_kawai(graph):
    length = 0.1
    k = 100.0
    eps = 0.1
    initial_temp = 1000

    alg_options = (eps, initial_temp)

    options = [100, 3]
    types = {1: 'barabasi_albert_graph', 2: 'complete_graph', 3: 'balanced_tree', 4: 'circular_ladder_graph',
             5: 'grid_graph', 6: 'grid_2d_graph', 7: 'hypercube_graph', 8: 'connected_caveman_graph'}

    graph = create_graph(options, graph_type=types[1])

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

    # plt.figure(figsize=(8, 8))
    # nx.draw_networkx(graph, coordinates, with_labels=True)
    # plt.show()
    degrees = list(graph.degree(graph.nodes()).values())

    print 'Started'
    new_coordinates = kamada_kawai_alg(graph, alg_options)
    print 'Ended'
    plt.figure(figsize=(8, 8))
    nx.draw_networkx(graph, new_coordinates, node_color=degrees, cmap=plt.cm.Blues, with_labels=False, node_size=100)
    plt.show()

    pos = nx.spring_layout(graph)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx(graph, pos, node_color=degrees, cmap=plt.cm.Blues, with_labels=False, node_size=100)
    plt.show()

    return 0


main()

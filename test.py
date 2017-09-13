#import requests
#with open('example.json', 'rb') as f:
#    r = requests.post('http://localhost:5000/graph', files={'example.json': f})


def choose_model(G, method='kamada-kawai', k=10000):
    if method == 'kamada-kawai':
        coords = kamada_kawai(G, k)
        return coords, 0
    elif method == 'fruchterman_reingold':
        coords = fruchterman_reingold(G, k)
        return coords, 0
    else:
        return jsonify({'error': 'Unknown method'}), -1


#def graph():
#    if not request.json or 'title' not in request.json:
#        abort(400)
#    else:
#        data = request.get_json(force=True)
#        G.add_nodes_from(data["Nodes"])
#        for arc in data["Arcs"]:
#            G.add_edge(arc["S"], arc["D"])
#        coords = nx.spring_layout(G)
#        return jsonify(data)
#@app.route('/graph', methods=['POST'])
#def graph():
#    return "Hello!"
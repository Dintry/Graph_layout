from app import app
from flask import jsonify
from flask import request
from flask import abort
import networkx as nx
from draw_method_new import kamada_kawai, fruchterman_reingold


@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"


@app.route('/layout', methods=['POST'])
def get_layout():
    if not request.json:
        abort(400)
    else:
        data = request.get_json(force=True)
        graph = nx.null_graph()
        for node in data[1]["Nodes"]:
            graph.add_node(node.values()[0])
        for arc in data[2]["Arcs"]:
            graph.add_edge(arc["S"], arc["D"])
        print data
        coordinates = kamada_kawai(graph)
        result = {u'Layout': []}
        for i in xrange(len(coordinates)):
            result[u'Layout'].append({u'X': coordinates[i][0], u'Y': coordinates[i][1]})
        data.append(result)
        return jsonify(data)

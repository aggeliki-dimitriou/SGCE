import nltk
from nltk.corpus import wordnet as wn
import json

def convert_to_graph(sgg):
    nodes = set()
    for triple in sgg:
      o, r, s = triple
      nodes.add(o)
      nodes.add(s)

    labels = {}
    labels_inverse = {}
    for id, label in enumerate(list(nodes)):
      labels[id] = label
      labels_inverse[label] = id

    nodes_ids = list(labels.keys())

    edges = []
    preds = {}
    for triple in sgg:
      o, r, s = triple
      edge = (labels_inverse[o], labels_inverse[s])
      edges.append(edge)
      preds[edge] = r

    syn_N = {}
    for k, v in labels.items():
      if wn.synsets(v):
        syn_N[k] = [wn.synsets(v)[0]]
      else:
        if len(v.split(' '))>1:
          syn_N[k] =  [wn.synsets(v.split(' ')[-1])[0]]
        else:
          if v == 'himself':
            syn_N[k] = [wn.synsets('self')[0]]
          elif v == 'that':
            syn_N[k] = [wn.synsets('thing')[0]]
          else:
            print(v)

    syn_E = {}
    for k, v in preds.items():
      if wn.synsets(v):
        syn_E[k] = [wn.synsets(v)[0]]
      else:
        syn_E[k] = []

    syn_E_ = {}
    preds_ = {}
    edges_ = []
    for k, v in syn_E.items():
      if v != []:
        syn_E_[k] = v
        preds_[k] = preds[k]
        edges_.append(k)

    return nodes_ids, edges_, labels, preds_, syn_N, syn_E_

def to_triples(sg):
  f = open(sg)
  caption_sgg = json.load(f)
  f.close()

  nodes = [i['key'] for i in caption_sgg['nodeDataArray'] if i['color'] != 'yellow']
  pairs = [(i['from'], i['to']) for i in caption_sgg['linkDataArray']]

  triples = []
  for i in range(len(pairs)):
    for j in range(i, len(pairs)):
      if pairs[i][1] == pairs[j][0] and pairs[i][1] not in nodes:
        triples.append([pairs[i][0], pairs[i][1], pairs[j][1]])

  return triples

def make_graph_from_json(scene_graph):

  nodes, edges, labels, preds, synN, synE = convert_to_graph(scene_graph)

  G = nx.DiGraph()
  # G.add_nodes_from(nodes)
  for node in nodes:
    G.add_node(node)
    G.nodes[node]['label'] = labels[node]
  for edge in edges:
    G.add_edge(edge[0], edge[1], label = preds[edge])

  return G, labels, preds, synN, synE

#f = open('relTD_relashionships.json') #located in drive driving vs parked/
#all_sggs = json.load(f)
#f.close()
#make_graph_from_json(to_triples(glob.glob('driving vs parked/caption-sgg/bicycle-driving/*.json')[30]))
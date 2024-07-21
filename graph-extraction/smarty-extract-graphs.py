import pickle as pkl
import json
import networkx as nx

def find_categories(word, size):
  return [k for k, v in tbox_dict.items() if word in v and len(v) == size]

tbox_dict = pkl.load(open('tbox_dict.pickle', 'rb'))

categories = {'characteristics': find_categories('User', 2),
              'conditions': find_categories('PreexistingCondition', 2),
              'sub-conditions': find_categories('PreexistingCondition', 3),
              'environment': find_categories('UserInstance', 2),
              'symptoms': find_categories('Symptom', 2),
              'sub-enviroment': find_categories('UserInstance', 3),
              'sub-symptoms': find_categories('Symptom', 3),
              'sub-sub-symptoms': find_categories('Symptom', 4)}

def create_smarty_graph_dicts_audible(data):
  objects = data['objects']

  nodes_dict = {0: 'User'}
  edges_dict = {}
  cnt = 0

  categories_ = {key: value for key, value in categories.items() if key not in ['conditions', 'sub-conditions', 'environment', 'sub-enviroment']}

  for o in objects:
    for cat, content in categories_.items():
      # print(o, cat, content)
      if o in content:
        cnt+=1
        nodes_dict[cnt] = o

        if 'sub-sub' in cat:
          node_2 = [thing for thing in tbox_dict[o] if thing in categories_['sub-symptoms']][0]
          if node_2 not in nodes_dict.values():
            cnt += 1
            nodes_dict[cnt] = node_2

          node_3 = [thing for thing in tbox_dict[o] if thing in categories_['symptoms']][0]
          if node_3 not in nodes_dict.values():
            cnt += 1
            nodes_dict[cnt] = node_3

          inverse_dict = {value: key for key, value in nodes_dict.items()}
          edges_dict[(0 , inverse_dict[node_3])] = 'symptoms'
          edges_dict[(inverse_dict[node_3], inverse_dict[node_2])] = 'sub-symptoms'
          edges_dict[(inverse_dict[node_2], inverse_dict[o])] = cat

        elif 'sub' in cat:
          node_2 = [thing for thing in tbox_dict[o] if thing not in ['owl.Thing', 'Symptom']][0]
          if node_2 not in nodes_dict.values():
            cnt += 1
            nodes_dict[cnt] = node_2

            inverse_dict = {value: key for key, value in nodes_dict.items()}
            edges_dict[(0, inverse_dict[node_2])] = cat.split('-')[1]
            edges_dict[(inverse_dict[node_2] , inverse_dict[o])] = cat

        else:
          edges_dict[(0, cnt)] = cat

  return nodes_dict, edges_dict

def make_graph_smarty(data, audible=False):

  if audible:
    nodes_dict, edges_dict = create_smarty_graph_dicts_audible(data)
  else:
    nodes_dict, edges_dict = create_smarty_graph_dicts(data)

  nodes = list(nodes_dict.keys())
  edges = list(edges_dict.keys())
  preds = edges_dict
  labels = nodes_dict

  G = nx.DiGraph()
  for node in nodes:
    G.add_node(node)
    G.nodes[node]['label'] = labels[node]
  for edge in edges:
    G.add_edge(edge[0], edge[1], label = preds[edge])

  return G, labels, preds

#data_per_patient = pkl.load(open('data_per_patient.pickle', 'rb')) #in drive
#make_graph_smarty(data_per_patient['32f346de-1b76-473e-b37d-8b6b26b2975e'])
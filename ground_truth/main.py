import pickle as pkl
import dgl
import numpy as np
import networkx as nx
import nltk
from nltk.corpus import wordnet as wn
import itertools
from tqdm import tqdm
from ged import graph_edit_distance
import glob

dir_path = 'ged_large500_astar_*'

# Credits for calculation of concept distance code to CECE
# Author: George Filandrianos
# GitHub: https://github.com/geofila
# Date: 2022-07-11

def create_tbox(scene_graphs200, materialize = False):

    all_labels = set()
    for idx in range(len(scene_graphs200)):
      for node in list(scene_graphs200[idx].nodes()):
            lbl = scene_graphs200[idx].nodes()[node]['label'].lower()
            all_labels.add(lbl)

    for idx in range(len(scene_graphs200)):
      for edge in list(scene_graphs200[idx].edges()):
            lbl_edge = scene_graphs200[idx].edges()[edge]['label'].lower()
            all_labels.add(lbl_edge)

    ## the tbox (dictionary: key is concept, value is definition (list of concepts))
    tbox = {c:set() for c in all_labels}
    ## connect with wordnet
    for c in tbox:
        syns = wn.synsets(c)
        if len(syns)>0:
            tbox[c].add(wn.synsets(c)[0].name())
        else:
            syns = wn.synsets(c.replace(' ',''))
            if len(syns)>0:
                tbox[c].add(wn.synsets(c.replace(' ',''))[0].name())
            else:
                wrds = c.split(' ')
                for w in wrds:
                    if len(wn.synsets(w))>0:
                        tbox[c].add(wn.synsets(w)[0].name())

    if materialize:
        for c in tbox:
            while True:
                curr_len = len(tbox[c])
                new_syns = set()
                for syn in tbox[c]:
                    hypers = [c.name() for c in wn.synset(syn).hypernyms()]
                    new_syns = new_syns.union(hypers)
                tbox[c] = tbox[c].union(new_syns)
                if len(tbox[c])==curr_len:
                    break
    return tbox

def obj_distance(obj1, obj2):
    """"
    Τhe function of calculating the transition of each object to another.
    """
    lca = obj1.intersection(obj2)
    diffs = len(obj1 - lca) +  len(obj2 - lca)
    if diffs < 15:
        return diffs
    else:
        return 10e6


def get_graph_costs(G1, G2, G1_dgl, G2_dgl, tbox):
  nodes_1 = G1.nodes()
  nodes_2 = G2.nodes()

  edges_1 = G1.edges()
  edges_2 = G2.edges()

  sub_costs_nodes = np.zeros((len(nodes_1), len(nodes_2)))
  sub_costs_edges = np.zeros((len(edges_1), len(edges_2)))

  G1_node_deletion_cost = np.zeros(len(nodes_1))
  G1_edge_deletion_cost = np.zeros(len(edges_1))
  G2_node_insertion_cost = np.zeros(len(nodes_2))
  G2_edge_insertion_cost = np.zeros(len(edges_2))

  olist1 = sorted(list(nodes_1))
  olist2 = sorted(list(nodes_2))

  for i in nodes_1:
      G1_node_deletion_cost[olist1.index(i)] = len(tbox[nodes_1[i]['label'].lower()])
      for j in nodes_2:
          G2_node_insertion_cost[olist2.index(j)] = len(tbox[nodes_2[j]['label'].lower()])
          sub_costs_nodes[olist1.index(i), olist2.index(j)] = obj_distance(tbox[nodes_1[i]['label'].lower()],
                                                                           tbox[nodes_2[j]['label'].lower()])

  for i in edges_1:
      ren_i = (olist1.index(i[0]), olist1.index(i[1]))
      idx1 = G1_dgl.edge_ids(ren_i[0], ren_i[1])
      G1_edge_deletion_cost[idx1] = len(tbox[edges_1[i]['label'].lower()])
      for j in edges_2:
          ren_j = (olist2.index(j[0]), olist2.index(j[1]))
          idx2 = G2_dgl.edge_ids(ren_j[0], ren_j[1])
          G2_edge_insertion_cost[idx2] = len(tbox[edges_2[j]['label'].lower()])
          sub_costs_edges[idx1, idx2] = obj_distance(tbox[edges_1[i]['label'].lower()],
                                                     tbox[edges_2[j]['label'].lower()])

  return sub_costs_nodes, sub_costs_edges, G1_node_deletion_cost, G1_edge_deletion_cost, G2_node_insertion_cost, G2_edge_insertion_cost

def find_ten_most_similar(idx, geds):
  sim_keys = [key for key in geds.keys() if idx in key]
  similarities = []
  sim_idx = []
  for key in sim_keys:
    similarities.append(geds[key][0])
    if key[0] == idx:
      sim_idx.append(key[1])
    else:
      sim_idx.append(key[0])

  sorted_sim = list(np.argsort(similarities))[:10]
  most_sim_10 = [sim_idx[sim] for sim in sorted_sim]

  return most_sim_10


def find_all_ten(geds,idx_list):
  result = dict()
  for i in idx_list:
    result[i] = find_ten_most_similar(i, geds)

  return result
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    large_graphs = pkl.load(open("../data/scene_graphs500_large.pkl", "rb"))
    large_graphs_idx = pkl.load(open("../data/scene500_large_idx.pkl", "rb"))

    scene_graphs500_dgl = [dgl.from_networkx(g) for g in large_graphs]

    for i in range(len(large_graphs)):
        graph = large_graphs[i]
        self_edges = list(nx.selfloop_edges(graph))
        if (self_edges != []):
            print(i, self_edges)
            for j in self_edges:
                large_graphs[i].remove_edge(*j)

            scene_graphs500_dgl[i] = dgl.from_networkx(graph)

    tbox = create_tbox(large_graphs, True)

    geds_bipartite_costs = dict()

    pairs = list(itertools.combinations(scene_graphs500_dgl, 2))
    large_graphs_idx = [elem[0] for elem in large_graphs_idx]
    pair_idxs = list(itertools.combinations(large_graphs_idx, 2))
    pairs_nx = list(itertools.combinations(large_graphs, 2))


    files = glob.glob(dir_path)
    if(len(files) > 0):
        ckpnt = [int(file.split('_')[-1][:-4])for file in files]
        max_ckpnt = ckpnt.index(max(ckpnt))
        amount = ckpnt[max_ckpnt]
        #print(amount, files[max_ckpnt])
        pairs = pairs[amount:]
        pair_idxs = pair_idxs[amount:]
        pairs_nx = pairs_nx[amount:]

        geds_bipartite_costs = pkl.load(open(files[max_ckpnt], 'rb'))

        counter = amount

    for pair, pair_idx, pair_nx in tqdm(zip(pairs, pair_idxs, pairs_nx)):
      sub_costs_nodes, sub_costs_edges, G1_node_deletion_cost, G1_edge_deletion_cost, G2_node_insertion_cost, G2_edge_insertion_cost = get_graph_costs(pair_nx[0], pair_nx[1], pair[0], pair[1], tbox)
      geds_bipartite_costs[pair_idx] = graph_edit_distance(pair[0], pair[1], sub_costs_nodes, sub_costs_edges, G1_node_deletion_cost, G2_node_insertion_cost, G1_edge_deletion_cost,
                                                           G2_edge_insertion_cost, 'bipartite')

    pkl.dump(geds_bipartite_costs, open('../outs/geds_bipartite_costs.pkl', 'wb'))

    ten_most_similar_large = find_all_ten(geds_bipartite_costs, large_graphs_idx)

    pkl.dump(ten_most_similar_large, open('../outs/ten_most_similar_large_bipartite.pkl', 'wb'))


import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
import requests
from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram, PyramidMatch, PropagationAttr, SubgraphMatching, GraphHopper
import time
import pickle as pkl

def print_image(image_data, id):
  url = image_data[id]['url']
  response = requests.get(url, stream=True)
  img = Image.open(response.raw)

  # plt.imshow(img)
  # plt.show()
  return img

def prepare_graph_kernel(G, syn_n, embedding_map, syn_e = None):

  # create edge index from
  edge_set = set(G.edges())

  embeddings = {}

  for i in list(G.nodes()):
    names = syn_n[i]
    if len(names)!= 1:
      print("Multiple Synsets", names)
    #TODO
    emb = embedding_map[names[0]]
    if None in np.array(emb):
      print(emb)
    embeddings[i] = list(np.array(emb))

  edge_embeddings = {}
  if syn_e != None:
    for i in list(G.edges()):
      names = syn_e[i]
      if len(names)!= 1:
        print("Multiple Synsets", names)
      #TODO
      if names[0] in embedding_map.keys():
        emb = embedding_map[names[0]]
      else:
        emb = np.array([names[0]])

      if None in np.array(emb):
        print(emb)

      edge_embeddings[i] = list(emb)

  result = [edge_set, embeddings, edge_embeddings]
  return result

def find_embedding_glove(names, embedding_map):
  emb = []
  hypers = []
  for i in names:
    name = i.split('.')[0]
    try: 
      emb = embedding_map[name]
      break
    except:
      components = name.split('_')
      hypers = [c.name() for c in wn.synset(i).hypernyms()]
      for j in components:
        try: 
          emb = embedding_map[j]
          break
        except:
          continue 
    
  return emb, hypers

def prepare_graph_kernel_2(G, syn_n, embedding_map, syn_e = None):

  # create edge index from 
  edge_set = set(G.edges())

  embeddings = {}

  for i in list(G.nodes()):
    names = syn_n[i]
    if len(names)!= 1:
      print("Multiple Synsets", names)
    #TODO
    emb=[]
    while(emb == []):
      emb, names = find_embedding_glove(names, embedding_map)
    
    embeddings[i] = list(np.array(emb))

  edge_embeddings = {}
  if syn_e != None:
      for i in list(G.edges()):
        names = syn_e[i]
        if len(names)!= 1:
          print("Multiple Synsets", names)
        #TODO
        emb=[]
        while(emb == []):
          emb, names = find_embedding_glove(names, embedding_map)
        
        edge_embeddings[i] = list(emb)


  result = [edge_set, embeddings, edge_embeddings]
  return result

def most_similar_graph(idx, emb, ten=False):
  sorted_emb = list(np.argsort(emb[idx]))
  sorted_emb.remove(idx) # remove self
  if ten:
    sim = sorted_emb[-10:]
  else:
    sim = sorted_emb
  return sim


def find_all_similar(emb, ten = False):
  size = emb.shape[0]
  similarities = []
  for i in range(size):
    sim = most_similar_graph(i, emb, ten)
    similarities.append(sim)
  return similarities

def compute_kernel(gk, G_train, out_path, graph_idx):
    t = time.time()
    K_train = gk.fit_transform(G_train)
    print('Done! {} seconds elapsed.'.format(time.time()-t))

    pkl.dump(K_train, open(out_path + '_rank.pkl', 'wb'))
    sim500 = find_all_similar(K_train)

    for idx in range(500):
        sim500[idx] = [graph_idx[i][0] for i in sim500[idx] ]

    pkl.dump(sim500, open(out_path + '.pkl', 'wb'))
    return sim500

def main():
    #load graphs, attributes
    large_graphs = pkl.load(open('../data/scene_graphs500_large.pkl', 'rb'))    ###
    large_graphs_idx = pkl.load(open('../data/scene500_large_idx.pkl', 'rb'))   ###

    embeddings_glove = pkl.load( open( "../data/glove_emb_300.pkl", "rb" ) )    ###

    syn_n200_large = pkl.load(open('../data/syn_n500_large.pkl', 'rb'))         ###
    syn_e200_large = pkl.load(open('../data/syn_e500_large.pkl', 'rb'))         ###

    #compute kernels without attributes
    G_train = list(graph_from_networkx(large_graphs, node_labels_tag='label', edge_labels_tag='label'))

    #compute kernels with attributes
    G_train_attr= []
    for G, syn_N, syn_E in zip(large_graphs, syn_n200_large, syn_e200_large):
        G_iter = prepare_graph_kernel_2(G, syn_N, embeddings_glove, syn_E)
        G_train_attr.append(G_iter)

    gk1 = WeisfeilerLehman(n_iter=1, normalize=False, base_graph_kernel=VertexHistogram)
    gk2 = PyramidMatch()
    gk3 = PropagationAttr(normalize=False)
    gk4 = SubgraphMatching(normalize=False)
    gk5 = GraphHopper(normalize=False, kernel_type='linear')

    sim = []
    print('Computing Weifeiler-Lehman Kernel...')
    sim.append(compute_kernel(gk1, G_train, '../outs/sim500_wl', large_graphs_idx))
    # a = gk2.fit(G_train)
    # pkl.dump(a, open('hists_pm.pkl', 'wb'))
    print('Computing Pyramid Match Kernel...')
    sim.append(compute_kernel(gk2, G_train, '../outs/sim500_pm', large_graphs_idx))
    print('Computing Propagation Attribute Kernel...')
    sim.append(compute_kernel(gk3, G_train_attr, '../outs/sim500_pa_attr', large_graphs_idx))
    print('Computing Subgraph Matching Kernel...')
    sim.append(compute_kernel(gk4, G_train_attr, '../outs/sim500_sm_attr', large_graphs_idx))
    print('Computing GraphHopper Kernel...')
    sim.append(compute_kernel(gk5, G_train_attr, '../outs/sim500_gh_attr', large_graphs_idx))

    f = open('../data/image_data.json')
    image_data = json.load(f)
    f.close()

    #print for an image (id = 10)
    for j in range(len(sim)):
        f, axarr = plt.subplots(1,10,figsize=(30,30))
        for i in range(10):
            axarr[i].imshow(print_image(image_data, sim[j][10][9-i]))

if __name__ == "__main__":
    main()
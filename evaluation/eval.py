import numpy as np
from sklearn.metrics import ndcg_score
import pickle as pkl
import rbo
from tqdm import tqdm
from scipy import spatial
from sklearn.metrics.pairwise import pairwise_distances

##image_data.json.zip must be unzipped in its directory

amount=500

def find_most_similar(idx, geds):
  sim_keys = [key for key in geds.keys() if idx in key]
  similarities = []
  sim_idx = []
  for key in sim_keys:
    similarities.append(geds[key][0])
    if key[0] == idx:
      sim_idx.append(key[1])
    else:
      sim_idx.append(key[0])

  sorted_sim = list(np.argsort(similarities))
  most_sim_10 = [sim_idx[sim] for sim in sorted_sim]

  return most_sim_10


def find_all(geds,idx_list):
  result = dict()
  for i in idx_list:
    result[i] = find_most_similar(i, geds)

  return result

#most similar to 0
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

def precision(a, b):
  print(np.array(a).shape, np.array(b).shape)
  assert np.array(a).shape == np.array(b).shape
  intermediate_hit = []
  for i in range(len(a)):
    hits = set(a[i]).intersection(set(b[i]))
    hits_per = len(hits)/len(a[i])
    intermediate_hit.append(hits_per)

  return intermediate_hit, np.mean(intermediate_hit)

#most similar to 0
def most_similar_graph(idx, emb, ten=False):
  sorted_emb = list(np.argsort(emb[idx]))
  sorted_emb.remove(idx) # remove self
  if ten:
    sim = sorted_emb[-10:]
  else:
    sim = sorted_emb[0] 
  return sim


def find_all_similar(emb, ten = False):
  size = emb.shape[0]
  similarities = []
  for i in range(size):
    sim = most_similar_graph(i, emb, ten)
    similarities.append(sim)
  return similarities

def score_stats(gd, sims):
  hps = dict()
  rbos = dict()
  for k in [10, 5, 2]:
    hps[k] = list()
    rbos[k] = list()
    print('---- k = {} ----'.format(k))
    for sim in sims:
      all_hps, mean_hp = precision(np.array(sim)[:,:k], np.array(gd)[:,:k])
      hps[k].append(all_hps)
      print("Precision@k: {}".format(mean_hp))
      all_rbos = list()
      for i in range(amount):
        all_rbos.append(rbo.RankingSimilarity(gd[i][:k], sim[i][:k]).rbo())
      
      rbos[k].append(all_rbos)
      print("Mean RBO: {}".format(np.mean(all_rbos)))

  return hps, rbos

def format_ndcg(a, b):
  a_s = [len(a) - i for i in range(len(a))]
  score_dict = {key:val for key, val in zip(a, a_s)}
  b_s = [score_dict[i] if i in score_dict.keys() else 0 for i in b]

  return a_s, b_s

def compute_ndcg(true_rank, pred_rank, k):
  true_rank_s = []
  pred_rank_s = []
  for i,j in zip(true_rank, pred_rank):
    i_s, j_s = format_ndcg(i,j)
    true_rank_s.append(i_s)
    pred_rank_s.append(j_s)
 
  return ndcg_score(np.array(true_rank_s), np.array(pred_rank_s), k=k)

def format_ndcg_2(a, b):
  a_s = [0 for _ in range(len(a))]
  a_s[0] = 1
  score_dict = {key:val for key, val in zip(a, a_s)}
  b_s = [score_dict[i] if i in score_dict.keys() else 0 for i in b]

  return a_s, b_s

def compute_ndcg_2(true_rank, pred_rank, k):
  true_rank_s = []
  pred_rank_s = []
  for i,j in zip(true_rank, pred_rank):
    i_s, j_s = format_ndcg_2(i,j)
    true_rank_s.append(i_s)
    pred_rank_s.append(j_s)
 
  return ndcg_score(np.array(true_rank_s), np.array(pred_rank_s), k=k)

def precision_k(a, b, k):
  if b[0] in a[:k]:
    return 1 
  else: 
    return 0

def get_rank_target_class(rank, target, classes):
  rank_targets =[]
  for idx in rank:
    if classes[idx] == target:
      rank_targets.append(idx)
  return rank_targets

def get_rank_untargeted_class(index, rank, classes):
    rank_targets =[]
    query_class = classes[index]
    for idx in rank:
      if classes[idx] != query_class:
        rank_targets.append(idx)
    return rank_targets

def main():

    print('-- Starting Evaluation --')

    print()
    print('Loading data...', end=' ')
    try:
      new_idxs = pkl.load(open('../data/scene500_large_idx.pkl', 'rb'))
      vg_dense_classification = pkl.load(open('../data/vg_dense.pkl', 'rb'))
      geds = pkl.load(open('../outs/ged_large500_2.pkl', 'rb'))

    except:
      print('Provide all needed data in correct directory.')

    gd_rank = find_all(geds, [i[0] for i in new_idxs])

    gd = []
    for idx in new_idxs:
      gd.append(gd_rank[idx[0]])

    class_dict =  dict()
    for i in range(len(new_idxs)):
      class_dict[new_idxs[i][0]] = vg_dense_classification.iloc[[i]]['label'].item()

    preds_gd = {}
    for i in range(500):
      idx = new_idxs[i][0]
      preds_gd[idx] = get_rank_untargeted_class(idx, gd[i], class_dict)

    #Kernels rank of scores
    sims_rank = []
    try:
      sims_rank.append(pkl.load(open('../outs/sim_wl_rank.pkl', 'rb')))
      sims_rank.append(pkl.load(open('../outs/sim_pm_rank.pkl', 'rb')))
      sims_rank.append(pkl.load(open('../outs/sim500_pa_attr_rank.pkl', 'rb')))
      sims_rank.append(pkl.load(open('../outs/sim500_sm_attr_rank.pkl', 'rb')))
      sims_rank.append(pkl.load(open('../outs/sim500_gh_attr_rank.pkl', 'rb')))
    except:
      print('Provide kernel score ranks in directory "outs"!')

    comp = []
    for i in sims_rank:
      c = []
      for j in i:
        mj = max(j)
        new_j = j/mj
        c.append(new_j)

      comp.append(c)

    emb_10_list = []
    for i in range(len(comp)):
      emb_10 = find_all_similar(np.array(comp[i]))
      for idx in range(500):
        emb_10[idx] = [new_idxs[i][0] for i in emb_10[idx] ]
      emb_10 = [list(reversed(i)) for i in emb_10]
      emb_10_list.append(emb_10)

    all_preds = []
    for j in range(len(emb_10_list)):
      preds = []
      for i in range(500):
        idx = new_idxs[i][0]
        preds.append(get_rank_untargeted_class(idx, emb_10_list[j][i], class_dict)[:10])
      all_preds.append(preds)

    a, b = score_stats([preds_gd[idx[0]][:10] for idx in new_idxs], all_preds)

    print('NDCG@k')
    for k in [10,5,2]:
      print()
      print('--------{}--------'.format(k))
      for i in range(len(emb_10_list)):
        print(compute_ndcg([preds_gd[idx[0]][:10] for idx in new_idxs], all_preds[i], k=k))
    
    print('NDCG@k binary')
    for k in [10,5,2]:
      print()
      print('--------{}--------'.format(k))
      for i in range(len(emb_10_list)):
        print(compute_ndcg_2([preds_gd[idx[0]][:10] for idx in new_idxs], all_preds[i], k=k))

    print('p@k binary')
    for k in [10,5,2]:
      print()
      print('--------{}--------'.format(k))
      for i in range(len(emb_10_list)):
        print(np.mean([precision_k(b,c, k=k) for b,c in zip([preds_gd[idx[0]][:10] for idx in new_idxs], all_preds[i])]))

    #GNN embeddings
    embeddings_res = []
    try:
      embeddings_res.append(np.array(pkl.load(open('../outs/emb_gcn_best.pkl', 'rb'))))
      embeddings_res.append(np.array(pkl.load(open('../outs/emb_gat_best.pkl', 'rb'))))
      embeddings_res.append(np.array(pkl.load(open('../outs/emb_gin_best.pkl', 'rb'))))
    except:
      print('Provide embeddings in directory "outs"!')

    print('Done')
    print()
    print('Creating rankings from GNN embeddings...')

    emb_10 = []

    for emb in embeddings_res:
      comp = []
      for value in tqdm(range(emb.shape[0])):
        emb_comp = []
        for i in range(emb.shape[0]):
          emb_comp.append(1 - spatial.distance.cosine(emb[value], emb[i]))

        comp.append([i/max(emb_comp) for i in emb_comp])

      emb_10 = find_all_similar(np.array(comp))
      for idx in range(500):
        emb_10[idx] = [new_idxs[i][0] for i in emb_10[idx] ]

      emb_10 = [list(reversed(i)) for i in emb_10]

      #or use this identical function which is much faster
      #no reversing
      #comp = pairwise_distances(embeddings_res, metric='cosine')
      #comp = comp/ np.max(comp, axis=1, keepdims=True)
      #emb_10 = find_all_similar(np.array(comp))
      #for idx in range(500):
      #  emb_10[idx] = [new_idxs[i][0] for i in emb_10[idx] ]

      preds = {}
      for i in range(500):
        idx = new_idxs[i][0]
        preds[idx] = get_rank_untargeted_class(idx, emb_10[i], class_dict)

      a, b = score_stats([preds_gd[idx[0]][:10] for idx in new_idxs], [[preds[idx[0]][:10] for idx in new_idxs]])

      print('NDCG@k')
      for k in [10,5,2]:
        print()
        print('--------{}--------'.format(k))
        for i in range(len([emb_10])):
          print(compute_ndcg([preds_gd[idx[0]][:10] for idx in new_idxs], [preds[idx[0]][:10] for idx in new_idxs], k=k))

      print('NDCG@k binary')
      for k in [10,5,2]:
        print()
        print('--------{}--------'.format(k))
        for i in range(len([emb_10])):
          print(compute_ndcg_2([preds_gd[idx[0]][:10] for idx in new_idxs], [preds[idx[0]][:10] for idx in new_idxs], k=k))

      print('p@k binary')
      for k in [10,5,2]:
        print()
        print('--------{}--------'.format(k))
        for i in range(len([emb_10])):
          print(np.mean([precision_k(b,c, k=k) for b,c in zip([preds_gd[idx[0]][:10] for idx in new_idxs], [preds[idx[0]][:10] for idx in new_idxs])]))

if __name__ == "__main__":
    main()
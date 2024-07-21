#construct scene graph from SGG labels after performning SGG
#steps for SGG are available in tutorial linked in main paper
#steps after graph construction, for embedding and evaluation are identical as described in README.md for VG.

def extract_scene_graph(image_idx):
  ind_to_classes = custom_data_info['ind_to_classes']
  ind_to_predicates = custom_data_info['ind_to_predicates']

  box_scores = custom_prediction[str(image_idx)]['bbox_scores']
  box_labels = deepcopy(custom_prediction[str(image_idx)]['bbox_labels'])
  all_rel_labels = custom_prediction[str(image_idx)]['rel_labels']
  all_rel_scores = custom_prediction[str(image_idx)]['rel_scores']
  all_rel_pairs = custom_prediction[str(image_idx)]['rel_pairs']

  box_topk = len([i for i in box_scores if i > 0.2])

  for i in range(len(box_labels)):
      # print(type(i), type(box_labels[i]))
      box_labels[i] = ind_to_classes[box_labels[i]]

  rel_labels = []
  rel_scores = []
 
  def give_synset(string):
    syn = wn.synsets(string)
    if syn == []:
      return []
    else:
      return [wn.synsets(string)[0].name()]

  nodes = []
  edges = []
  node_labels = {}
  edge_labels = {}
  preds = {}
  rels ={}

  for i in range(len(all_rel_pairs)):
      if all_rel_pairs[i][0] < box_topk and all_rel_pairs[i][1] < box_topk and all_rel_scores[i] > 0.5:
          rel_scores.append(all_rel_scores[i])
          label = str(all_rel_pairs[i][0]) + '_' + box_labels[all_rel_pairs[i][0]] + ' => ' + ind_to_predicates[all_rel_labels[i]] + ' => ' + str(all_rel_pairs[i][1]) + '_' + box_labels[all_rel_pairs[i][1]]
          rel_labels.append(label)

          if give_synset(box_labels[all_rel_pairs[i][0]]) != []:
            nodes.append(all_rel_pairs[i][0])
            node_labels[all_rel_pairs[i][0]] = box_labels[all_rel_pairs[i][0]]
            preds[all_rel_pairs[i][0]] = give_synset(box_labels[all_rel_pairs[i][0]])
          
          if give_synset(box_labels[all_rel_pairs[i][1]]) != []:
            nodes.append(all_rel_pairs[i][1])
            node_labels[all_rel_pairs[i][1]] = box_labels[all_rel_pairs[i][1]]
            preds[all_rel_pairs[i][1]] = give_synset(box_labels[all_rel_pairs[i][1]])

          if all_rel_pairs[i][0] in nodes and all_rel_pairs[i][1] in nodes:
            if give_synset(ind_to_predicates[all_rel_labels[i]]) != []:
              edges.append((all_rel_pairs[i][0], all_rel_pairs[i][1]))
              edge_labels[(all_rel_pairs[i][0], all_rel_pairs[i][1])] = ind_to_predicates[all_rel_labels[i]]
              rels[(all_rel_pairs[i][0], all_rel_pairs[i][1])] = give_synset(ind_to_predicates[all_rel_labels[i]])

  return list(set(nodes)), edges, preds, rels, node_labels, edge_labels


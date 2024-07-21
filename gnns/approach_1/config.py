model_type = 'gcn'                 #gat/gin: GNN variant for similarity component
learning_rate = 0.04               #for Adam
weight_decay = 0                   #for Adam
epochs = 50
batch_size = 32
K = 70000                          #training pairs (denoted as p in paper)
dims = [2048]                      #output dimensions for GNN embedding components, add more elements in list for more than 1 layers)
p = 0                              #dropout rate
training = False

g1 = 0                             #set only if trying to extract similarity score from GNN
g2 = 0

GRAPH_PATH = '../../data/scene_graphs500_large.pkl'          #subset of graphs (this corresponds to VG-DENSE)
TEST_GRAPH_PATH = ''                                         #set only if testing on unseen graphs
TEST_SYN_N_PATH = ''
SYN_N_PATH = '../../data/syn_n500_large.pkl'                 #synsets for nodes, load this file as an example for structure
SYN_E_PATH = '../../data/syn_e500_large.pkl'                 #synsets for edges
EMBEDDING_PATH = '../../../drive/MyDrive/glove_emb_300.pkl'  #list of embeddings to use for initialization
GED_PATH = '../../outs/geds_bipartite_costs.pkl'             #pre-computed GEDS
IDX_PATH = '../../data/scene500_large_idx.pkl'               #indexes of chosen subset graphs
LOAD_PATH = ''                                               #set only, if already trained model exists
EMB_SAVE_PATH = '../../outs/emb2.pkl'                        #output path for graph embeddings
SAVE_PATH = '../../outs/model_1.pth'                         #output path for trained model
SIM = ''                                                     #set only if trying to extract similarity score from GNN


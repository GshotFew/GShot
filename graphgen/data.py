import pickle
from torch.utils.data import Dataset

from dfscode.dfs_wrapper import get_min_dfscode
from datasets.preprocess import dfscode_to_tensor
import torch, numpy as np
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.seed_all()

class Graph_DFS_code_from_file(Dataset):
    """
    Dataset for reading graphs from files and returning matrices
    corresponding to dfs code entries
    :param args: Args object
    :param graph_list: List of graph indices to be included in the dataset
    :param feature_map: feature_map for the dataset generated by the mapping
    """

    def __init__(self, dict_dataset_current, graph_list, feature_map):
        # Path to folder containing dataset
        self.dataset_path = dict_dataset_current['current_processed_dataset_path']#args.current_processed_dataset_path
        self.graph_list = graph_list
        self.feature_map = feature_map
        self.temp_path = dict_dataset_current['current_temp_path']

        self.max_edges = feature_map['max_edges']
        max_nodes, len_node_vec, len_edge_vec = feature_map['max_nodes'], len(
            feature_map['node_forward']) + 1, len(feature_map['edge_forward']) + 1
        self.feature_len = 2 * (max_nodes + 1) + 2 * \
            len_node_vec + len_edge_vec

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        with open(self.dataset_path + 'graph' + str(self.graph_list[idx]) + '.dat', 'rb') as f:
            dfscode_tensors = pickle.load(f)

        return dfscode_tensors


class Graph_DFS_code(Dataset):
    """
    Mainly for testing purposes
    Dataset for returning matrices corresponding to dfs code entries
    :param args: Args object
    :param graph_list: List of graph indices to be included in the dataset
    :param feature_map: feature_map for the dataset generated by the mapping
    """

    def __init__(self, args, graph_list, feature_map):
        # Path to folder containing dataset
        self.graph_list = graph_list
        self.feature_map = feature_map
        self.temp_path = args.current_temp_path

        self.max_edges = feature_map['max_edges']
        max_nodes, len_node_vec, len_edge_vec = feature_map['max_nodes'], len(
            feature_map['node_forward']) + 1, len(feature_map['edge_forward']) + 1
        self.feature_len = 2 * (max_nodes + 1) + 2 * \
            len_node_vec + len_edge_vec

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        G = self.graph_list[idx]

        # Get DFS code matrix
        min_dfscode = get_min_dfscode(G, self.temp_path)
        print(min_dfscode)

        # dfscode tensors
        dfscode_tensors = dfscode_to_tensor(min_dfscode, self.feature_map)

        return dfscode_tensors
import random
import time
import pickle
from torch.utils.data import DataLoader

from base_args import Args
from utils import create_dirs
from datasets.process_dataset import create_graphs, get_mapping, gen_dfs_code
from datasets.preprocess import calc_max_prev_node, dfscodes_weights
from baselines.graph_rnn.data import Graph_Adj_Matrix_from_file
from graphgen.data import Graph_DFS_code_from_file
from model import create_model
from train import train, train_meta, train_multi, fine_tune
import sys
import torch, numpy as np
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.seed_all()
import pdb

if __name__ == '__main__':
    arg_type = sys.argv[1]
    args = Args(arg_type)
    #args = args.update_args()
    print(vars(args))
    create_dirs(args)

    DATASETS = args.DATASETS
    dict_dataset_all = {}
    
    DICT_GRAPHS_NUMBER_EACH_TYPE = {}
    DICT_GRAPHS_NUMBER_EACH_TYPE['NUM_TRAIN_GRAPHS'] = {}
    DICT_GRAPHS_NUMBER_EACH_TYPE['NUM_VAL_GRAPHS'] = {}
    DICT_GRAPHS_NUMBER_EACH_TYPE['NUM_TEST_GRAPHS'] = {}

    for dataset,cts in args.train_val_dict.items():
        DICT_GRAPHS_NUMBER_EACH_TYPE['NUM_TRAIN_GRAPHS'][dataset] = cts[0]
        DICT_GRAPHS_NUMBER_EACH_TYPE['NUM_VAL_GRAPHS'][dataset] = cts[1]
        DICT_GRAPHS_NUMBER_EACH_TYPE['NUM_TEST_GRAPHS'][dataset] = cts[2]
        

    print(" DICT_GRAPHS_NUMBER_EACH_TYPE ", DICT_GRAPHS_NUMBER_EACH_TYPE)
    

    for dataset in DATASETS:
        print("dataset ", dataset)
        args.graph_type=dataset
        args.num_graphs = DICT_GRAPHS_NUMBER_EACH_TYPE['NUM_TRAIN_GRAPHS'][dataset] + DICT_GRAPHS_NUMBER_EACH_TYPE['NUM_VAL_GRAPHS'][dataset] + DICT_GRAPHS_NUMBER_EACH_TYPE['NUM_TEST_GRAPHS'][dataset] # TOTAL_GRAPHS
        print("Num of graphs", args.num_graphs)
        dict_dataset_current = create_graphs(args)
        dict_dataset_all[dataset] = dict_dataset_current

    feature_map = get_mapping(dict_dataset_all, multiple_datsets_on= True,args=args)
    print('feature_map', feature_map)
    
    
    for dataset in DATASETS:
        print(" new dataset ", dataset)
        args.graph_type = dataset
        dict_dataset_all[dataset]['feature_map'] = feature_map
        if args.note == 'DFScodeRNN':
            print(' gen dfs code')
            gen_dfs_code(dict_dataset_all[dataset])

        graphs = dict_dataset_all[dataset]['graphs']

        random.shuffle(graphs)

        NUM_TRAIN_GRAPHS = DICT_GRAPHS_NUMBER_EACH_TYPE['NUM_TRAIN_GRAPHS'][dataset]
        NUM_VAL_GRAPHS = DICT_GRAPHS_NUMBER_EACH_TYPE['NUM_VAL_GRAPHS'][dataset]
        
        graphs_train = graphs[: int(NUM_TRAIN_GRAPHS)]

        USE_FOR_TRAINING = NUM_TRAIN_GRAPHS

        print("USE_FOR_TRAINING ", USE_FOR_TRAINING)
        graphs_train = graphs_train[:USE_FOR_TRAINING]

        graphs_validate = graphs[int(NUM_TRAIN_GRAPHS): NUM_TRAIN_GRAPHS + NUM_VAL_GRAPHS ]#+ int(0.90 * len(graphs))]

        print(" graphs_train ", len(graphs_train))
        print(" graphs_validate ", len(graphs_validate))
        
        # show graphs statistics
        print('Model:', args.note)
        print('Device:', args.device)
        print('Graph type:', args.graph_type)
        print('Training set: {}, Validation set: {}'.format(
            len(graphs_train), len(graphs_validate)))

        # Loading the feature map
        print(" current_dataset_path ", args.current_dataset_path)

        print('Max number of nodes: {}'.format(feature_map['max_nodes']))
        print('Max number of edges: {}'.format(feature_map['max_edges']))
        print('Min number of nodes: {}'.format(feature_map['min_nodes']))
        print('Min number of edges: {}'.format(feature_map['min_edges']))
        print('Max degree of a node: {}'.format(feature_map['max_degree']))
        print('No. of node labels: {}'.format(len(feature_map['node_forward'])))
        print('No. of edge labels: {}'.format(len(feature_map['edge_forward'])))

        if args.note == 'GraphRNN':
            start = time.time()
            if args.max_prev_node is None:
                args.max_prev_node = calc_max_prev_node(
                    dict_dataset_all[dataset]['current_processed_dataset_path'])

            dict_dataset_all[dataset]['max_prev_node'] = args.max_prev_node

            args.max_head_and_tail = None
            dict_dataset_all[dataset]['max_head_and_tail'] = args.max_head_and_tail

            print('max_prev_node:', dict_dataset_all[dataset]['max_prev_node'])  # args.max_prev_node)

            end = time.time()
            print('Time taken to calculate max_prev_node = {:.3f}s'.format(
                end - start))

        if args.note == 'DFScodeRNN' and args.weights:
            feature_map = {
                **feature_map,
                **dfscodes_weights(args.min_dfscode_path, graphs_train, feature_map, args.device)
            }

        if args.note == 'GraphRNN':
            random_bfs = True
            dataset_train = Graph_Adj_Matrix_from_file(
                dict_dataset_all[dataset], graphs_train, feature_map, random_bfs)
            dataset_validate = Graph_Adj_Matrix_from_file(
                dict_dataset_all[dataset], graphs_validate, feature_map, random_bfs)
            #
        else:

            dataset_train = Graph_DFS_code_from_file(
                dict_dataset_all[dataset], graphs_train, feature_map)
            dataset_validate = Graph_DFS_code_from_file(
                dict_dataset_all[dataset], graphs_validate, feature_map)        
        

        dataloader_train = DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.num_workers)
        dataloader_validate = DataLoader(
            dataset_validate, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)

        dict_dataset_all[dataset]['dataloader_train'] = dataloader_train
        dict_dataset_all[dataset]['dataloader_validate'] = dataloader_validate
        dict_dataset_all[dataset]['feature_map'] = feature_map
        dict_dataset_all[dataset]['training'] = True

    print("creating model")
    model = create_model(args, feature_map)

    dict_dataset_all[args.target_dataset]['training'] =False
    
    train_multi(args, dict_dataset_all, model)

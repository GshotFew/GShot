import random
import time
import pickle
from torch.utils.data import DataLoader

#from args import Args
from base_args import Args
from utils import create_dirs
from datasets.process_dataset import create_graphs, get_mapping, gen_dfs_code
from datasets.preprocess import calc_max_prev_node, dfscodes_weights
from baselines.graph_rnn.data import Graph_Adj_Matrix_from_file
from graphgen.data import Graph_DFS_code_from_file
from model import create_model
from train import train, train_meta, train_multi
import sys
import torch, numpy as np
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.seed_all()

if __name__ == '__main__':
    args = Args(sys.argv[1])
    #args = args.update_args()


    create_dirs(args)

    random.seed(123)

    dict_dataset_all = {}

    DATASETS = args.DATASETS  

    NUM_TRAIN_GRAPHS =args.train_val_dict[DATASETS[0]][0]
    NUM_VAL_GRAPHS = args.train_val_dict[DATASETS[0]][1]
    NUM_TEST_GRAPHS = args.train_val_dict[DATASETS[0]][2]

    TOTAL_GRAPHS = NUM_TRAIN_GRAPHS +NUM_VAL_GRAPHS + NUM_TEST_GRAPHS

    for dataset in DATASETS:
        print("dataset ", dataset)
        args.graph_type=dataset
        args.num_graphs = TOTAL_GRAPHS
        print(" args ", args.graph_type, args.fname)

        dict_dataset_current = create_graphs(args)
        dict_dataset_all[dataset] = dict_dataset_current

    feature_map = get_mapping(dict_dataset_all,multiple_datsets_on= False,args=args)

    print(" args ", args.graph_type, args.fname)

    for dataset in DATASETS:
        print(" new dataset ", dataset)
        dict_dataset_all[dataset]['feature_map'] = feature_map

        if args.note == 'DFScodeRNN':
            print(' gen dfs code')
            gen_dfs_code(dict_dataset_all[dataset])

        print(" feature_map ", feature_map)
        graphs = dict_dataset_all[dataset]['graphs']

        random.seed(123)
        random.shuffle(graphs)

        graphs_train = graphs[: int(NUM_TRAIN_GRAPHS)]

        if(args.USE_FOR_TRAINING is not None):
            USE_FOR_TRAINING = args.USE_FOR_TRAINING
            graphs_train = graphs_train[:USE_FOR_TRAINING]

        print("USE_FOR_TRAINING ", args.USE_FOR_TRAINING)

        graphs_validate = graphs[int(NUM_TRAIN_GRAPHS): NUM_TRAIN_GRAPHS + NUM_VAL_GRAPHS ]#+ int(0.90 * len(graphs))]

        print(" graphs_train ", len(graphs_train))
        print(" graphs_validate ", len(graphs_validate))

        # show graphs statistics
        print('Model:', args.note)
        print('Device:', args.device)
        print('Graph type:', args.graph_type)
        print('Training set: {}, Validation set: {}'.format(
            len(graphs_train), len(graphs_validate)))



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

            dict_dataset_all[dataset]['max_prev_node'] = args.max_prev_node
            args.max_head_and_tail = None
            dict_dataset_all[dataset]['max_head_and_tail'] = args.max_head_and_tail

            print('max_prev_node:', dict_dataset_all[dataset]['max_prev_node'])# args.max_prev_node)

            end = time.time()
            print('Time taken to calculate max_prev_node = {:.3f}s'.format(
                end - start))
        #

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


        print(' args.batch_size ', args.batch_size)

        dataloader_train = DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.num_workers)
        dataloader_validate = DataLoader(
            dataset_validate, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)

        dict_dataset_all[dataset]['dataloader_train'] = dataloader_train
        dict_dataset_all[dataset]['dataloader_validate'] = dataloader_validate
        dict_dataset_all[dataset]['feature_map'] = feature_map

    

    model = create_model(args, feature_map)

    print('args.milestones', args.milestones)
    train(args, dict_dataset_all, model)

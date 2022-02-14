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

if __name__ == '__main__':
    args = Args(sys.argv[1])
    args = args.update_args() ### To set produce graphs, mindfscode etc to False

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
        #args = Args()
        args.graph_type=dataset
        args.num_graphs = DICT_GRAPHS_NUMBER_EACH_TYPE['NUM_TRAIN_GRAPHS'][dataset] + DICT_GRAPHS_NUMBER_EACH_TYPE['NUM_VAL_GRAPHS'][dataset]+DICT_GRAPHS_NUMBER_EACH_TYPE['NUM_TEST_GRAPHS'][dataset]# TOTAL_GRAPHS

        args.produce_graphs = False
        # Whether to produce min dfscode and write to files
        args.produce_min_dfscodes = False
        # Whether to map min dfscodes to tensors and save to files
        args.produce_min_dfscode_tensors = False


        print(" args ", args.graph_type, args.fname)


        dict_dataset_current = create_graphs(args)
        dict_dataset_all[dataset] = dict_dataset_current

    #feature_map = get_mapping(dict_dataset_all)
    print("loading from dict")
    f = open(args.current_model_save_path+"/map.dict","rb")
    feature_map = pickle.load(f)
    f.close()
    
    args.load_model = True
    # args.epochs_save = 100
    args.epochs_save = 20
    args.epochs_validate = 20
    args.epochs = 20000

    args.load_model_path = 'model_save/{}_'.format(args.note)+args.run_type+"_"+args.time+"/"+'{}_'.format(args.note) +args.run_type+"_"+str(args.load_epoch_tune)+".dat"
    print("Loading this model", args.load_model_path)

    print(" args ", args.graph_type, args.fname)

    for dataset in DATASETS:
        args.graph_type = dataset

        print(" new dataset ", dataset)
        dict_dataset_all[dataset]['feature_map'] = feature_map
        #gen_dfs_code(dict_dataset_all[dataset])

        print(" feature_map ", feature_map)
        graphs = dict_dataset_all[dataset]['graphs']

        random.seed(123)
        random.shuffle(graphs)
        # graphs_train = graphs[: int(0.10 * len(graphs))]

        NUM_TRAIN_GRAPHS = DICT_GRAPHS_NUMBER_EACH_TYPE['NUM_TRAIN_GRAPHS'][dataset]
        NUM_VAL_GRAPHS = DICT_GRAPHS_NUMBER_EACH_TYPE['NUM_VAL_GRAPHS'][dataset]

        graphs_train = graphs[: int(NUM_TRAIN_GRAPHS)]

        USE_FOR_TRAINING = args.USE_FOR_TRAINING #NUM_TRAIN_GRAPHS# -30##-250 ### set it in config

        print("USE_FOR_TRAINING ", USE_FOR_TRAINING)
        print('len(USE_FOR_TRAINING) ',USE_FOR_TRAINING)

        graphs_train = graphs_train[:USE_FOR_TRAINING]

        graphs_validate = graphs[int(NUM_TRAIN_GRAPHS): NUM_TRAIN_GRAPHS + NUM_VAL_GRAPHS ]#+ int(0.90 * len(graphs))]



        print(" graphs_train ", graphs_train)
        print(" graphs_validate ", graphs_validate)

        # show graphs statistics
        print('Model:', args.note)
        print('Device:', args.device)
        print('Graph type:', args.graph_type)
        print('Training set: {}, Validation set: {}'.format(
            len(graphs_train), len(graphs_validate)))

        # Loading the feature map
        # print(" current_dataset_path ", args.current_dataset_path)
        # with open(args.current_dataset_path + 'map.dict', 'rb') as f:
        #     feature_map = pickle.load(f)

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



        if (args.target_dataset == 'springs_n_balls_5'):
            args.batch_size =  500

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
        dict_dataset_all[dataset]['training'] = False


    model = create_model(args, feature_map)

    dict_dataset_all[args.target_dataset]['training'] =True
   
    target_dataset = args.target_dataset#'MUTAG'
    

    fine_tune(args, dict_dataset_all, target_dataset, model)



import os
import random
import shutil
from statistics import mean
import torch
from base_args import Args

from graphgen.train import predict_graphs as gen_graphs_dfscode_rnn
from baselines.graph_rnn.train import predict_graphs as gen_graphs_graph_rnn
from utils import get_model_attribute, load_graphs, save_graphs
import torch, numpy as np
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.seed_all()
import metrics.stats
import pdb 
import sys
import pickle
LINE_BREAK = '----------------------------------------------------------------------\n'


class ArgsEvaluate():
    def __init__(self):
        # Can manually select the device too
        args = Args(sys.argv[1])

        self.device = torch.device(
            'cuda:{}'.format(str(args.GPU)) if torch.cuda.is_available() else 'cpu')
        if args.is_meta:
            self.model_path = 'model_save/'+str(args.note)+'_'+args.run_type+"_"+args.time+"/"+str(args.note)+'_'+args.run_type+"_"+str(args.load_epoch_eval)+".dat"+args.target_dataset
           
        else:
           
            self.model_path = 'model_save/'+str(args.note)+'_'+args.run_type+"_"+args.time+"/" +str(args.note)+'_'+args.run_type+"_"+str(args.load_epoch_eval)+".dat"
        print("Model path to be evaluated", self.model_path)
        self.num_epochs = get_model_attribute(
            'epoch', self.model_path, self.device)

        # Whether to generate networkx format graphs for real datasets
        self.generate_graphs = True

        self.count = 1024
        self.batch_size =10

        self.metric_eval_batch_size =10

        # Specific DFScodeRNN
        print("loading from dict")
        f = open(args.current_model_save_path+"/map.dict","rb")
        feature_map = pickle.load(f)
        f.close()
        self.max_num_edges = feature_map['local_feature_map'][args.target_dataset]['max_edges']   ### set it in config
        self.max_num_node = feature_map['local_feature_map'][args.target_dataset]['max_nodes']    ### set it in config

        #graphrnn
        self.min_num_node = feature_map['local_feature_map'][args.target_dataset]['min_nodes']

        print("Max no. edges, nodes",self.max_num_edges,self.max_num_node)

        self.train_args = get_model_attribute(
            'saved_args', self.model_path, self.device)

        self.graphs_save_path = 'graphs/'
        self.current_graphs_save_path = self.graphs_save_path + self.train_args.fname + '_' + \
                self.train_args.time + '/' + str(self.num_epochs) + '/'

            

        self.num_eval_train = args.num_eval_train
        self.num_eval_test = args.num_eval_test
def patch_graph(graph):
    for u in graph.nodes():
        graph.nodes[u]['label'] = graph.nodes[u]['label'].split('-')[0]

    return graph


def generate_graphs(eval_args):
    """
    Generate graphs (networkx format) given a trained generative model
    and save them to a directory
    :param eval_args: ArgsEvaluate object
    """

    train_args = eval_args.train_args

    if train_args.note == 'GraphRNN':
        gen_graphs = gen_graphs_graph_rnn(eval_args)
    elif train_args.note == 'DFScodeRNN':
        gen_graphs = gen_graphs_dfscode_rnn(eval_args)
    elif train_args.note == 'DGMG':
        gen_graphs = gen_graphs_dgmg(eval_args)

    if os.path.isdir(eval_args.current_graphs_save_path):
        shutil.rmtree(eval_args.current_graphs_save_path)

    os.makedirs(eval_args.current_graphs_save_path)

    save_graphs(eval_args.current_graphs_save_path, gen_graphs)


def print_stats(

    node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref,
    edge_count_avg_pred, degree_mmd, clustering_mmd, orbit_mmd,
    nspdk_mmd, node_label_mmd, edge_label_mmd, node_label_and_degree
):
    print('Node count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(node_count_avg_ref), mean(node_count_avg_pred)))
    print('Edge count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(edge_count_avg_ref), mean(edge_count_avg_pred)))

    print('MMD Degree - {:.6f}, MMD Clustering - {:.6f}, MMD Orbits - {:.6f}'.format(
        mean(degree_mmd), mean(clustering_mmd), mean(orbit_mmd)))
    print('MMD NSPDK - {:.6f}'.format(mean(nspdk_mmd)))
    print('MMD Node label - {:.6f}, MMD Edge label - {:.6f}'.format(
        mean(node_label_mmd), mean(edge_label_mmd)
    ))
    print('MMD Joint Node label and degree - {:.6f}'.format(
        mean(node_label_and_degree)
    ))
    print(LINE_BREAK)


if __name__ == "__main__":
    eval_args = ArgsEvaluate()
    train_args = eval_args.train_args
    print("Current dataset path", train_args.current_dataset_path)
    print("Model path", eval_args.model_path)
    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    if eval_args.generate_graphs:
        generate_graphs(eval_args)

    random.seed(123)
    #pdb.set_trace()
    graphs = []
    for name in os.listdir(train_args.current_dataset_path):
        if name.endswith('.dat'):
            graphs.append(len(graphs))

    random.shuffle(graphs)

    graphs_train_indices = graphs[0:eval_args.num_eval_train]  #### train 
    graphs_test_indices =graphs[eval_args.num_eval_test:]   # train+val:




    graphs_pred_indices = []
    if not eval_args.generate_graphs:
        for name in os.listdir(eval_args.current_graphs_save_path):
            if name.endswith('.dat'):
                graphs_pred_indices.append(len(graphs_pred_indices))
    else:
        graphs_pred_indices = [i for i in range(eval_args.count)]

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    print('Graphs generated - {}'.format(len(graphs_pred_indices)))

    metrics.stats.novelity(
        train_args.current_dataset_path, graphs_train_indices, eval_args.current_graphs_save_path,
        graphs_pred_indices, train_args.temp_path, timeout=60)

    metrics.stats.uniqueness(
        eval_args.current_graphs_save_path,
        graphs_pred_indices, train_args.temp_path, timeout=120)

    # exit()

    node_count_avg_ref, node_count_avg_pred = [], []
    edge_count_avg_ref, edge_count_avg_pred = [], []

    degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd = [], [], [], []
    node_label_mmd, edge_label_mmd, node_label_and_degree = [], [], []
    
    graphs_ref = load_graphs(
            train_args.current_dataset_path, graphs_test_indices)
    node_count_avg_ref.append(mean([len(G.nodes()) for G in graphs_ref]))
    
    #print("TESTING")
    #print(node_count_avg_ref)
    #print(mean([len(G.edges()) for G in graphs_ref]))
    #print("TESTING DONE")

    #print(len(graphs_test_indices))
    #print(graphs_test_indices)
    for i in range(0, len(graphs_pred_indices), eval_args.metric_eval_batch_size):

        #ahil


        batch_size = min(eval_args.metric_eval_batch_size,
                         len(graphs_pred_indices) - i)

        print(" batch_size ", batch_size)
        
        graphs_ref_indices = random.sample(graphs_test_indices, batch_size)
        print(graphs_ref_indices)
        #break
        graphs_ref = load_graphs(
            train_args.current_dataset_path, graphs_ref_indices)

        graphs_ref = [patch_graph(g) for g in graphs_ref]

        graphs_pred = load_graphs(
            eval_args.current_graphs_save_path, graphs_pred_indices[i: i + batch_size])

        graphs_pred = [patch_graph(g) for g in graphs_pred]

        node_count_avg_ref.append(mean([len(G.nodes()) for G in graphs_ref]))
        node_count_avg_pred.append(mean([len(G.nodes()) for G in graphs_pred]))

        print(node_count_avg_ref)
        edge_count_avg_ref.append(mean([len(G.edges()) for G in graphs_ref]))
        edge_count_avg_pred.append(mean([len(G.edges()) for G in graphs_pred]))
        #pdb.set_trace()
        degree_mmd.append(metrics.stats.degree_stats(graphs_ref, graphs_pred))
        clustering_mmd.append(
            metrics.stats.clustering_stats(graphs_ref, graphs_pred))
        orbit_mmd.append(metrics.stats.orbit_stats_all(
            graphs_ref, graphs_pred))

        nspdk_mmd.append(metrics.stats.nspdk_stats(graphs_ref, graphs_pred))

        node_label_mmd.append(
            metrics.stats.node_label_stats(graphs_ref, graphs_pred))
        edge_label_mmd.append(
            metrics.stats.edge_label_stats(graphs_ref, graphs_pred))
        node_label_and_degree.append(
            metrics.stats.node_label_and_degree_joint_stats(graphs_ref, graphs_pred))

        print('Running average of metrics:\n')

        print_stats(
            node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
            degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd, node_label_mmd,
            edge_label_mmd, node_label_and_degree
        )

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    print_stats(
        node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
        degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd, node_label_mmd,
        edge_label_mmd, node_label_and_degree
    )
    metrics.stats.novelity(
        train_args.current_dataset_path, graphs_train_indices, eval_args.current_graphs_save_path,
        graphs_pred_indices, train_args.temp_path, timeout=60)

    metrics.stats.uniqueness(
        eval_args.current_graphs_save_path,
        graphs_pred_indices, train_args.temp_path, timeout=120)
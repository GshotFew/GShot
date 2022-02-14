import os
import random
import time
import math
import pickle
from functools import partial
from multiprocessing import Pool
import bisect
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
from scipy import stats
import shutil
from utils import mkdir
from datasets.preprocess import (
    mapping, graphs_to_min_dfscodes,
    min_dfscodes_to_tensors, random_walk_with_restart_sampling
)


def check_graph_size(
    graph, min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):

    if min_num_nodes and graph.number_of_nodes() < min_num_nodes:
        return False
    if max_num_nodes and graph.number_of_nodes() > max_num_nodes:
        return False

    if min_num_edges and graph.number_of_edges() < min_num_edges:
        return False
    if max_num_edges and graph.number_of_edges() > max_num_edges:
        return False

    return True


def produce_graphs_from_raw_format(
    inputfile, output_path, num_graphs=None, min_num_nodes=None,
    max_num_nodes=None, min_num_edges=None, max_num_edges=None
):
    """
    :param inputfile: Path to file containing graphs in raw format
    :param output_path: Path to store networkx graphs
    :param num_graphs: Upper bound on number of graphs to be taken
    :param min_num_nodes: Lower bound on number of nodes in graphs if provided
    :param max_num_nodes: Upper bound on number of nodes in graphs if provided
    :param min_num_edges: Lower bound on number of edges in graphs if provided
    :param max_num_edges: Upper bound on number of edges in graphs if provided
    :return: number of graphs produced
    """

    lines = []
    with open(inputfile, 'r') as fr:
        for line in fr:
            line = line.strip().split()
            lines.append(line)

    index = 0
    count = 0
    graphs_ids = set()
    while index < len(lines):
        if lines[index][0][1:] not in graphs_ids:
            graph_id = lines[index][0][1:]
            G = nx.Graph(id=graph_id)

            index += 1
            vert = int(lines[index][0])
            index += 1
            for i in range(vert):
                G.add_node(i, label=lines[index][0])
                index += 1

            edges = int(lines[index][0])
            index += 1
            for i in range(edges):
                G.add_edge(int(lines[index][0]), int(
                    lines[index][1]), label=lines[index][2])
                index += 1

            index += 1

            if not check_graph_size(
                G, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
            ):
                continue

            if nx.is_connected(G):
                with open(os.path.join(
                        output_path, 'graph{}.dat'.format(count)), 'wb') as f:
                    pickle.dump(G, f)

                graphs_ids.add(graph_id)
                count += 1

                if num_graphs and count >= num_graphs:
                    break

        else:
            vert = int(lines[index + 1][0])
            edges = int(lines[index + 2 + vert][0])
            index += vert + edges + 4

    return count

def produce_graphs_from_raw_format_active(
    inputfile, output_path, num_graphs=None, min_num_nodes=None,
    max_num_nodes=None, min_num_edges=None, max_num_edges=None, active_file_path=None
):
    """
    :param inputfile: Path to file containing graphs in raw format
    :param output_path: Path to store networkx graphs
    :param num_graphs: Upper bound on number of graphs to be taken
    :param min_num_nodes: Lower bound on number of nodes in graphs if provided
    :param max_num_nodes: Upper bound on number of nodes in graphs if provided
    :param min_num_edges: Lower bound on number of edges in graphs if provided
    :param max_num_edges: Upper bound on number of edges in graphs if provided
    :return: number of graphs produced
    """

    dict_active_molecules={}

    with open(active_file_path, 'r') as fr:
        for line in fr:
            line = line.strip()

            if('#' in line ): #leukemia
                graph_id = line[1:]
            else:
                graph_id = line[0:] #aids-ca

            dict_active_molecules[graph_id]=1


            # lines.append(line)


    lines = []
    with open(inputfile, 'r') as fr:
        for line in fr:
            line = line.strip().split()
            lines.append(line)


    #print(' dict_active_molecules' ,dict_active_molecules, len(dict_active_molecules))

    index = 0
    count = 0
    graphs_ids = set()
    while index < len(lines):
        if lines[index][0][1:] not in graphs_ids:
            graph_id = lines[index][0][1:]

            # if(graph_id not in dict_active_molecules):
            #     print('skippping ',graph_id , ' inactive' )
            #     continue



            G = nx.Graph(id=graph_id)

            index += 1
            vert = int(lines[index][0])
            index += 1
            for i in range(vert):
                G.add_node(i, label=lines[index][0])
                index += 1

            edges = int(lines[index][0])
            index += 1
            for i in range(edges):
                G.add_edge(int(lines[index][0]), int(
                    lines[index][1]), label=lines[index][2])
                index += 1

            #exception AIDSca
            if('AIDS-ca' not in inputfile):
                index += 1

            if (graph_id not in dict_active_molecules):
                # print('skippping ', graph_id, ' inactive')
                continue
            else:
                pass
                #print('active ', graph_id)



            if not check_graph_size(
                G, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
            ):
                continue

            if nx.is_connected(G):
                with open(os.path.join(
                        output_path, 'graph{}.dat'.format(count)), 'wb') as f:
                    pickle.dump(G, f)

                graphs_ids.add(graph_id)
                #print('inserted ',graph_id)
                count += 1

                if num_graphs and count >= num_graphs:
                    break

        else:
            vert = int(lines[index + 1][0])
            edges = int(lines[index + 2 + vert][0])
            index += vert + edges + 4

    return count


# For Enzymes dataset
def produce_graphs_from_graphrnn_format(
    input_path, dataset_name, output_path, num_graphs=None,
    node_invariants=[], min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None, dict_node_label_mapping=None
):
    node_attributes = False
    graph_labels = False
    edge_labels = True

    G = nx.Graph()
    # load data
    path = input_path
    print("dataset_name, ", dataset_name)
    print(" path , ", path)
    data_adj = np.loadtxt(os.path.join(path, dataset_name + '_A.txt'),
                          delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(
            os.path.join(path, dataset_name + '_node_attributes.txt'),
            delimiter=',')

    data_node_label = np.loadtxt(
        os.path.join(path, dataset_name + '_node_labels.txt'),
        delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(
        os.path.join(path, dataset_name + '_graph_indicator.txt'),
        delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(
            os.path.join(path, dataset_name + '_graph_labels.txt'),
            delimiter=',').astype(int)

    if edge_labels:
        data_edge_labels = np.loadtxt(
            os.path.join(path, dataset_name + '_edge_labels.txt'),
            delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)
    # G.edges[2,1]
    for edge_labels_index, edge_type  in enumerate(data_edge_labels):
        pair_edge = data_adj[edge_labels_index]
        G.edges[pair_edge]['label'] = str(edge_type)


    # add node labels
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i + 1, feature=data_node_att[i])
        G.add_node(i + 1, label=str(data_node_label[i]))

    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1

    count = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['id'] = data_graph_labels[i]

        if not check_graph_size(
            G_sub, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
        ):
            continue

        if nx.is_connected(G_sub):
            G_sub = nx.convert_node_labels_to_integers(G_sub)
            G_sub.remove_edges_from(nx.selfloop_edges(G_sub))

            if 'CC' in node_invariants:
                clustering_coeff = nx.clustering(G_sub)
                cc_bins = [0, 0.2, 0.4, 0.6, 0.8]

            for node in G_sub.nodes():
                node_label = str(G_sub.nodes[node]['label'])

                if(dict_node_label_mapping is not None):
                    node_label = dict_node_label_mapping[node_label]
                    # print("remapped node label")

                if 'Degree' in node_invariants:
                    node_label += '-' + str(G_sub.degree[node])

                if 'CC' in node_invariants:
                    node_label += '-' + str(
                        bisect.bisect(cc_bins, clustering_coeff[node]))

                G_sub.nodes[node]['label'] = node_label

            with open(os.path.join(
                    output_path, 'graph{}.dat'.format(count)), 'wb') as f:
                pickle.dump(G_sub, f)

            count += 1

            if num_graphs and count >= num_graphs:
                break

    return count



# For Enzymes dataset
def produce_graphs_from_graphrnn_format_Enzyme_classes(
    input_path, dataset_name, output_path, num_graphs=None,
    node_invariants=[], min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None, target_class_file_path=None
):
    node_attributes = False
    graph_labels = True
    with open(target_class_file_path, 'r') as fr:
        target_class = int(fr.read().strip())

    print(' target_class ', target_class)

    G = nx.Graph()
    # load data
    path = input_path
    data_adj = np.loadtxt(os.path.join(path, dataset_name + '_A.txt'),
                          delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(
            os.path.join(path, dataset_name + '_node_attributes.txt'),
            delimiter=',')

    data_node_label = np.loadtxt(
        os.path.join(path, dataset_name + '_node_labels.txt'),
        delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(
        os.path.join(path, dataset_name + '_graph_indicator.txt'),
        delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(
            os.path.join(path, dataset_name + '_graph_labels.txt'),
            delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)

    # add node labels
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i + 1, feature=data_node_att[i])
        G.add_node(i + 1, label=str(data_node_label[i]))

    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1

    count = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['id'] = data_graph_labels[i]

            if(not(int(data_graph_labels[i])  == target_class)):
                continue


        if not check_graph_size(
            G_sub, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
        ):
            continue

        if nx.is_connected(G_sub):
            G_sub = nx.convert_node_labels_to_integers(G_sub)
            G_sub.remove_edges_from(nx.selfloop_edges(G_sub))

            if 'CC' in node_invariants:
                clustering_coeff = nx.clustering(G_sub)
                cc_bins = [0, 0.2, 0.4, 0.6, 0.8]

            for node in G_sub.nodes():
                node_label = str(G_sub.nodes[node]['label'])

                if 'Degree' in node_invariants:
                    node_label += '-' + str(G_sub.degree[node])

                if 'CC' in node_invariants:
                    node_label += '-' + str(
                        bisect.bisect(cc_bins, clustering_coeff[node]))

                G_sub.nodes[node]['label'] = node_label

            nx.set_edge_attributes(G_sub, 'DEFAULT_LABEL', 'label')

            with open(os.path.join(
                    output_path, 'graph{}.dat'.format(count)), 'wb') as f:
                pickle.dump(G_sub, f)

            count += 1

            if num_graphs and count >= num_graphs:
                break

    return count



def produce_physics_graphs(
    input_edge_path,input_node_feature_path, output_path, num_graphs=None, min_num_nodes=None,
    max_num_nodes=None, min_num_edges=None, max_num_edges=None
):
    """
    :param inputfile: Path to file containing graphs in raw format
    :param output_path: Path to store networkx graphs
    :param num_graphs: Upper bound on number of graphs to be taken
    :param min_num_nodes: Lower bound on number of nodes in graphs if provided
    :param max_num_nodes: Upper bound on number of nodes in graphs if provided
    :param min_num_edges: Lower bound on number of edges in graphs if provided
    :param max_num_edges: Upper bound on number of edges in graphs if provided
    :return: number of graphs produced
    """

    loc_train = np.load(input_node_feature_path)
    edges_train = np.load(input_edge_path)
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])

    # loc_min = loc_train.min(axis=(1, 2), keepdims=True)
    # loc_max = loc_train.max(axis=(1, 2), keepdims=True)

    loc_max,loc_min = loc_train.max(axis=(0, 1, 2)), loc_train.min(axis=(0, 1, 2))
    # loc_max = loc_train.max()
    # loc_min = loc_train.min()

    loc_train = (loc_train - loc_min)  / (loc_max - loc_min)








    count = 0
    graphs_ids = set()
    
    print('lenedges_train', len(edges_train))
    for graph_id, edge_adj_matrix in enumerate(edges_train):

        if num_graphs and count > num_graphs:
            break

        num_step =loc_train[graph_id].shape[1]
        #numtrain 100
        #length 1000
        # sample freq 50

        # binx = [0.0,0.25, 0.5, 0.75]#, 0.8]
        binx = np.linspace(0, 1.0, num=5)# [0.0,0.25, 0.5, 0.75]#, 0.8]
        biny = np.linspace(0, 1.0, num=5)  #
        steps_array= list(range(0,num_step))
        # steps_to_sample = 10
        # steps_to_use = random.sample(steps_array, steps_to_sample)
        # print('steps_to_use ', steps_to_use)
        # random.sample

        # for step in steps_to_use: #range(0,num_step):
        for step in range(0,num_step):
            G = nx.from_numpy_matrix(edge_adj_matrix)
            for node_id in G.nodes():

                G.nodes[node_id]['node_feature'] =loc_train[graph_id][node_id][step]# [1.5, 2.5]

                ret = stats.binned_statistic_2d([loc_train[graph_id][node_id][step][0]], [loc_train[graph_id][node_id][step][1]], None, 'count', bins=[binx, biny])
                bin_number = ret.binnumber[0]
                G.nodes[node_id]['label'] = str(bin_number)

            G.remove_edges_from(nx.selfloop_edges(G))
            pass
            nx.set_edge_attributes(G, 'DEFAULT_LABEL', 'label')

            if num_graphs and count > num_graphs:
                break


            if not check_graph_size(
                    G, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
            ):
                continue

            if nx.is_connected(G):
                with open(os.path.join(
                        output_path, 'graph{}.dat'.format(count)), 'wb') as f:
                    pickle.dump(G, f)



                graphs_ids.add(graph_id)
                count += 1




    # lines = []
    # with open(inputfile, 'r') as fr:
    #     for line in fr:
    #         line = line.strip().split()
    #         lines.append(line)
    #
    # index = 0
    # count = 0
    # graphs_ids = set()
    # while index < len(lines):
    #     if lines[index][0][1:] not in graphs_ids:
    #         graph_id = lines[index][0][1:]
    #         G = nx.Graph(id=graph_id)
    #
    #         index += 1
    #         vert = int(lines[index][0])
    #         index += 1
    #         for i in range(vert):
    #             G.add_node(i, label=lines[index][0])
    #             index += 1
    #
    #         edges = int(lines[index][0])
    #         index += 1
    #         for i in range(edges):
    #             G.add_edge(int(lines[index][0]), int(
    #                 lines[index][1]), label=lines[index][2])
    #             index += 1
    #
    #         index += 1
    #
    #         if not check_graph_size(
    #             G, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
    #         ):
    #             continue
    #
    #         if nx.is_connected(G):
    #             with open(os.path.join(
    #                     output_path, 'graph{}.dat'.format(count)), 'wb') as f:
    #                 pickle.dump(G, f)
    #
    #             graphs_ids.add(graph_id)
    #             count += 1
    #
    #             if num_graphs and count >= num_graphs:
    #                 break
    #
    #     else:
    #         vert = int(lines[index + 1][0])
    #         edges = int(lines[index + 2 + vert][0])
    #         index += vert + edges + 4

    return count


def sample_subgraphs(
    idx, G, output_path, iterations, num_factor, min_num_nodes=None,
    max_num_nodes=None, min_num_edges=None, max_num_edges=None
):
    count = 0
    deg = G.degree[idx]
    for _ in range(num_factor * int(math.sqrt(deg))):
        G_rw = random_walk_with_restart_sampling(
            G, idx, iterations=iterations, max_nodes=max_num_nodes,
            max_edges=max_num_edges)
        G_rw = nx.convert_node_labels_to_integers(G_rw)
        G_rw.remove_edges_from(nx.selfloop_edges(G_rw))

        if not check_graph_size(
            G_rw, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
        ):
            continue

        if nx.is_connected(G_rw):
            with open(os.path.join(
                    output_path,
                    'graph{}-{}.dat'.format(idx, count)), 'wb') as f:
                pickle.dump(G_rw, f)
                count += 1


def produce_random_walk_sampled_graphs(
    input_path, dataset_name, output_path, iterations, num_factor,
    num_graphs=None, min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):
    print('Producing random_walk graphs - num_factor - {}'.format(num_factor))
    G = nx.Graph()

    d = {}
    count = 0
    with open(os.path.join(input_path, dataset_name + '.content'), 'r') as f:
        for line in f.readlines():
            spp = line.strip().split('\t')
            G.add_node(count, label=spp[-1])
            d[spp[0]] = count
            count += 1

    count = 0
    with open(os.path.join(input_path, dataset_name + '.cites'), 'r') as f:
        for line in f.readlines():
            spp = line.strip().split('\t')
            if spp[0] in d and spp[1] in d:
                G.add_edge(d[spp[0]], d[spp[1]], label='DEFAULT_LABEL')
            else:
                count += 1

    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.convert_node_labels_to_integers(G)

    with Pool(processes=48) as pool:
        for _ in tqdm(pool.imap_unordered(partial(
                sample_subgraphs, G=G, output_path=output_path,
                iterations=iterations, num_factor=num_factor,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges),
                list(range(G.number_of_nodes())))):
            pass

    filenames = []
    for name in os.listdir(output_path):
        if name.endswith('.dat'):
            filenames.append(name)

    random.shuffle(filenames)

    if not num_graphs:
        num_graphs = len(filenames)

    count = 0
    for i, name in enumerate(filenames[:num_graphs]):
        os.rename(
            os.path.join(output_path, name),
            os.path.join(output_path, 'graph{}.dat'.format(i))
        )
        count += 1

    for name in filenames[num_graphs:]:
        os.remove(os.path.join(output_path, name))

    return count


# Routine to create datasets
def create_graphs(args):
    # Different datasets
    if 'Lung' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'Lung/')
        input_path = base_path + 'lung.txt'
        min_num_nodes, max_num_nodes = None, 50
        min_num_edges, max_num_edges = None, None

    elif 'Breast' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'Breast/')
        input_path = base_path + 'breast.txt'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'Leukemia' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'Leukemia/')
        input_path = base_path + '123.txt_graph'
        active_file_path = base_path + 'actives.txt'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'AIDS-ca' in args.graph_type:
        base_path = os.path.join(args.dataset_path, 'AIDS-ca/')
        input_path = base_path + 'aido99_all.txt'
        active_file_path = base_path + 'ca.txt'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'AIDS-cm' in args.graph_type:
        base_path = os.path.join(args.dataset_path, 'AIDS-cm/')
        input_path = base_path + 'cm.txt'
        # active_file_path = base_path + 'actives.txt'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None



    elif 'Yeast' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'Yeast/')
        input_path = base_path + 'yeast.txt'
        min_num_nodes, max_num_nodes = None, 50
        min_num_edges, max_num_edges = None, None

    # elif 'springs' == args.graph_type:
    elif 'spring' in args.graph_type:
        # n_balls = args.n_balls
        base_path = os.path.join(args.dataset_path, args.graph_type)+'/'#, 'springs_n_balls_'+str(n_balls))+'/'
        input_edge_path = base_path + 'edges_train_springs.npy'#.format(n_balls)
        input_node_feature_path = base_path + 'loc_train_springs.npy'#.format(n_balls)
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'charged' in args.graph_type:
        # n_balls = args.n_balls
        base_path = os.path.join(args.dataset_path, args.graph_type)+'/'#, 'springs_n_balls_'+str(n_balls))+'/'
        input_edge_path = base_path + 'edges_train_charged.npy'#.format(n_balls)
        input_node_feature_path = base_path + 'loc_train_charged.npy'#.format(n_balls)
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None


    elif 'All' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'All/')
        input_path = base_path + 'all.txt'
        # No limit on number of nodes and edges
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    # elif 'ENZYMES' in args.graph_type:
    #     base_path = os.path.join(args.dataset_path, 'ENZYMES/')
    #     # Node invariants - Options 'Degree' and 'CC'
    #     node_invariants = ['Degree']
    #     min_num_nodes, max_num_nodes = None, None
    #     min_num_edges, max_num_edges = None, None

    elif 'ENZYMES' in args.graph_type:
        base_path = os.path.join(args.dataset_path, args.graph_type+'/')#'ENZYMES/')
        target_class_file_path = base_path + 'myclas.txt'
        print(' base_path ', base_path)
        # Node invariants - Options 'Degree' and 'CC'
        node_invariants = ['Degree']
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None


    elif 'citeseer' in args.graph_type:
        base_path = os.path.join(args.dataset_path, 'citeseer/')
        random_walk_iterations = 150  # Controls size of graph
        num_factor = 5  # Controls size of dataset

        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = 20, None

    elif 'cora' in args.graph_type:
        base_path = os.path.join(args.dataset_path, 'cora/')
        random_walk_iterations = 150  # Controls size of graph
        num_factor = 5  # Controls size of dataset

        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = 20, None

    elif 'MUTAG' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'MUTAG/')
        # input_path = base_path + 'leukemia.txt'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    else:
        print('Dataset - {} is not valid'.format(args.graph_type))
        exit()

    args.current_dataset_path = os.path.join(base_path, args.run_type,'graphs/')
    args.min_dfscode_path = os.path.join(base_path, args.run_type,'min_dfscodes/')
    min_dfscode_tensor_path = os.path.join(base_path,args.run_type, 'min_dfscode_tensors/')

    if args.note == 'GraphRNN' or args.note == 'DGMG':
        args.current_processed_dataset_path = args.current_dataset_path
    elif args.note == 'DFScodeRNN':
        args.current_processed_dataset_path = min_dfscode_tensor_path

    if args.produce_graphs:
        #print("args ", args)
        #print("creating produce graphs ")
        mkdir(args.current_dataset_path)

        if args.graph_type in ['Lung', 'Breast',  'Yeast', 'All']:
        # if args.graph_type in ['Lung', 'Breast', 'Leukemia_GraphGen', 'Yeast', 'All']:
            count = produce_graphs_from_raw_format(
                input_path, args.current_dataset_path, args.num_graphs,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges)

        # elif args.graph_type in ['ENZYMES']:
        #     count = produce_graphs_from_graphrnn_format_Enzyme_classes(
        #         base_path, args.graph_type, args.current_dataset_path,
        #         num_graphs=args.num_graphs, node_invariants=node_invariants,
        #         min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
        #         min_num_edges=min_num_edges, max_num_edges=max_num_edges)

        elif 'ENZYMES' in args.graph_type:
            count = produce_graphs_from_graphrnn_format_Enzyme_classes(
                # base_path, args.graph_type, args.current_dataset_path,
                base_path, 'ENZYMES', args.current_dataset_path,
                num_graphs=args.num_graphs, node_invariants=node_invariants,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges, target_class_file_path=target_class_file_path)

        elif args.graph_type in ['cora', 'citeseer']:
            count = produce_random_walk_sampled_graphs(
                base_path, args.graph_type, args.current_dataset_path,
                num_graphs=args.num_graphs, iterations=random_walk_iterations,
                num_factor=num_factor, min_num_nodes=min_num_nodes,
                max_num_nodes=max_num_nodes, min_num_edges=min_num_edges,
                max_num_edges=max_num_edges)

        elif args.graph_type in ['MUTAG']:
            dict_node_label_mapping = {'0':'C', '1':'N', '2':'O', '3':'F', '4':'I', '5':'Cl', '6':'Br'}
            # dict_edge_label_mapping = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
            count = produce_graphs_from_graphrnn_format(
                base_path, args.graph_type, args.current_dataset_path,
                num_graphs=args.num_graphs,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges, dict_node_label_mapping=dict_node_label_mapping)


        elif args.graph_type in ['Leukemia']:
            count = produce_graphs_from_raw_format_active(
                input_path, args.current_dataset_path, args.num_graphs,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges, active_file_path=active_file_path)


        # elif args.graph_type in ['AIDS-ca''AIDS-ca']:
        elif 'AIDS-ca' in args.graph_type : #in ['AIDS-ca''AIDS-ca']:
            count = produce_graphs_from_raw_format_active(
                input_path, args.current_dataset_path, args.num_graphs,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges, active_file_path=active_file_path)

        if args.graph_type in ['AIDS-cm']:
            # if args.graph_type in ['Lung', 'Breast', 'Leukemia_GraphGen', 'Yeast', 'All']:
            count = produce_graphs_from_raw_format(
                input_path, args.current_dataset_path, args.num_graphs,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges)


        elif 'spring' in args.graph_type or 'charged' in args.graph_type:
            count = produce_physics_graphs(
                input_edge_path, input_node_feature_path, args.current_dataset_path, args.num_graphs,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges)

       ## print('Graphs produced', count)
    else:
        count = len([name for name in os.listdir(
            args.current_dataset_path) if name.endswith(".dat")])
        #print('Graphs counted', count)

    graphs = [i for i in range(count)]

    dict_dataset_current = {}

    dict_dataset_current['base_path'] = base_path
    dict_dataset_current['current_dataset_path'] = args.current_dataset_path
    dict_dataset_current['min_dfscode_path'] = args.min_dfscode_path
    dict_dataset_current['min_dfscode_tensor_path'] = min_dfscode_tensor_path
    dict_dataset_current['current_processed_dataset_path'] = args.current_processed_dataset_path
    dict_dataset_current['dataset_path'] = args.dataset_path
    dict_dataset_current['current_temp_path'] = args.current_temp_path
    dict_dataset_current['graph_type'] = args.graph_type
    dict_dataset_current['dataset_path'] = args.dataset_path
    dict_dataset_current['graphs'] =graphs




    return dict_dataset_current# graphs
    # return count

    #
def get_mapping(dict_dataset_current,multiple_datsets_on=False,args=None):
    feature_map = mapping(dict_dataset_current, multiple_datsets_on,args)

    return feature_map
    #later


    # # Produce feature map
    # feature_map = mapping(args.current_dataset_path,
    #                       args.current_dataset_path + 'map.dict')
    # print(feature_map)
def gen_dfs_code(dict_dataset_current):
    #print("dict_dataset_current ", dict_dataset_current)
    # base_path = os.path.join(dict_dataset_current['dataset_path'],
    #                          dict_dataset_current['graph_type'] + '/')
    base_path = os.path.join(dict_dataset_current['dataset_path'], dict_dataset_current['graph_type']+'/')

    # min_dfscode_tensor_path = os.path.join(base_path, 'min_dfscode_tensors/')
    min_dfscode_tensor_path = dict_dataset_current['min_dfscode_tensor_path']# ['os.path.join(base_path, 'min_dfscode_tensors/')



    # if dict_dataset_current['produce_min_dfscodes']:
        # Empty the directory
    if os.path.exists(dict_dataset_current['min_dfscode_path']):
        shutil.rmtree(dict_dataset_current['min_dfscode_path'])
    if not os.path.exists(dict_dataset_current['min_dfscode_path']):
        mkdir(dict_dataset_current['min_dfscode_path'])
        start = time.time()
        graphs_to_min_dfscodes(dict_dataset_current['current_dataset_path'],
                               dict_dataset_current['min_dfscode_path'], dict_dataset_current['current_temp_path'])

        end = time.time()
        print('Time taken to make dfscodes = {:.3f}s'.format(end - start))

    else:
        print("not gen min dfs code",dict_dataset_current['min_dfscode_path'])

    # if dict_dataset_current['produce_min_dfscode_tensors']:
        # Empty the directory
    # if not os.path.exists(min_dfscode_tensor_path):
    mkdir(min_dfscode_tensor_path)

    start = time.time()
    min_dfscodes_to_tensors(dict_dataset_current['min_dfscode_path'],
                            min_dfscode_tensor_path, dict_dataset_current['feature_map'])

    end = time.time()
    print('Time taken to make dfscode tensors= {:.3f}s'.format(
        end - start))

    # else:
    #     print("not gen min dfs code tensors ", min_dfscode_tensor_path)

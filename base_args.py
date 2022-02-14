from datetime import datetime
import torch
from utils import get_model_attribute
import numpy as np
import importlib
print("Import done from config")
class Args:
    """
    Program configuration
    """

    def __init__(self,config_type='None'):
        # Can manually select the device too


        # Clean tensorboard
        self.clean_tensorboard = False
        # Clean temp folder
        self.clean_temp = False

        # Whether to use tensorboard for logging
        self.log_tensorboard = True

        # Algorithm Version - # Algorithm Version - GraphRNN  | DFScodeRNN (GraphGen) | DGMG (Deep GMG)
        self.note = 'DFScodeRNN'
        
        # Check datasets/process_dataset for datasets
        # Select dataset to train the model
        # self.graph_type = 'FINA-META-YEAST-LUNG-BREAST-MODEL'
        print("Importing config", config_type)
        config = importlib.import_module('configs.{}'.format(config_type))


        self.graph_type = config.graph_type

        if('note' in dir(config)):
            self.note=config.note
            print('note is ', self.note)
            
        if('threshold' in dir(config)):
            self.threshold=config.threshold
            print('threshold is set by config  ', self.threshold)
        else:
            self.threshold=10000000000
            print('threshold is now set to  ', self.threshold)
            
        if('growing_factor' in dir(config)):
            self.growing_factor=config.growing_factor
            print('growing_factor is set by config as ', self.growing_factor)
            
        else:
            
            self.growing_factor=1
            print('growing_factor is now set to  ', self.growing_factor)

        self.num_graphs = 100  #None  # Set it None to take complete dataset

        # Whether to produce networkx format graphs for real datasets
        self.produce_graphs = True
        # Whether to produce min dfscode and write to files
        self.produce_min_dfscodes = True
        # Whether to map min dfscodes to tensors and save to files
        self.produce_min_dfscode_tensors = True

        # if none, then auto calculate
        self.max_prev_node = None  # max previous node that looks back for GraphRNN

        if ('max_prev_node' in dir(config)):
            self.max_prev_node = config.max_prev_node
            print('max_prev_node is ', self.max_prev_node)



        # Specific to GraphRNN
        # Model parameters
        self.hidden_size_node_level_rnn = 128  # hidden size for node level RNN
        self.embedding_size_node_level_rnn = 64  # the size for node level RNN input
        self.embedding_size_node_output = 64  # the size of node output embedding
        self.hidden_size_edge_level_rnn = 16  # hidden size for edge level RNN
        self.embedding_size_edge_level_rnn = 8  # the size for edge level RNN input
        self.embedding_size_edge_output = 8  # the size of edge output embedding

        # Specific to DFScodeRNN
        # Model parameters
        self.hidden_size_dfscode_rnn = 256  # hidden size for dfscode RNN
        self.embedding_size_dfscode_rnn = 92  # input size for dfscode RNN
        # the size for vertex output embedding
        self.embedding_size_timestamp_output = 512
        self.embedding_size_vertex_output = 512  # the size for vertex output embedding
        self.embedding_size_edge_output = 512  # the size for edge output embedding


        self.dfscode_rnn_dropout = 0.2  # Dropout layer in between RNN layers
        self.loss_type = 'BCE'
        self.weights = False

        # Specific to DFScodeRNN
        self.rnn_type = 'LSTM'  # LSTM | GRU

        # Specific to GraphRNN | DFScodeRNN
        self.num_layers = 4  # Layers of rnn

        # self.batch_size = 32  # normal: 32, and the rest should be changed accordingly
        self.batch_size = 32 # normal: 32, and the rest should be changed accordingly

        # Specific to DGMG
        # Model parameters
        self.feat_size = 128
        self.hops = 2
        self.dropout = 0.2

        # training config
        self.num_workers = 8  # num workers to load data, default 4
        self.epochs = 10000

        # self.lr = 0.003  # Learning rate
        self.lr = 0.003  # Learning rate
        # Learning rate decay factor at each milestone (no. of epochs)
        self.gamma = 0.3



        self.milestones = [100, 200, 400, 800]  # List of milestones
        self.schedule=True

        # Whether to do gradient clipping
        self.gradient_clipping = True

        # Output config
        self.dir_input = ''
        self.model_save_path = self.dir_input + 'model_save/'
        self.tensorboard_path = self.dir_input + 'tensorboard/'
        self.dataset_path = self.dir_input + 'datasets/'
        self.temp_path = self.dir_input + 'tmp/'

        # Model save and validate parameters
        self.save_model = True
        self.epochs_save = config.epochs_save
        self.epochs_validate = config.epochs_validate

        # Time at which code is run
        self.time = config.time  #'{0:%Y-%m-%d %H:%M:%S}'.format(datetime.now())
        self.run_type = config.run_type
        # Filenames to save intermediate and final outputs
        self.fname = self.note + '_' + self.run_type

        # Calcuated at run time
        self.current_model_save_path = self.model_save_path + \
            self.fname + '_' + self.time + '/'
        self.current_dataset_path = None
        self.current_processed_dataset_path = None
        self.current_min_dfscode_path = None
        self.current_temp_path = self.temp_path + self.fname + '_' + self.time + '/'

        # Model load parameters
        self.load_model = False
        self.load_model_path = ''
        self.device = torch.device(
            'cuda:{}'.format(config.GPU) if torch.cuda.is_available() else 'cpu')
        self.load_device = torch.device('cuda:{}'.format(config.GPU))
        self.epochs_end = config.epochs_end
        self.epochs = config.epochs_end


        self.uniqueness_reg=0

        self.DATASETS = config.DATASETS
        self.target_dataset = config.target_dataset
        self.train_val_dict = config.train_val_dict
        self.weight_decay = config.weight_decay

        if ('weight_decay_meta_train' in dir(config)):
            self.weight_decay_meta_train = config.weight_decay_meta_train
            print('weight_decay_meta_train is ', self.weight_decay_meta_train)

        #self.weight_decay_meta_train = config.weight_decay_meta_train

        self.alpha = config.alpha
        self.inner_steps = config.inner_steps
        self.USE_FOR_TRAINING = config.USE_FOR_TRAINING
        self.load_epoch_tune = config.load_epoch_tune
        self.GPU = config.GPU
        self.load_epoch_eval = config.load_epoch_eval
        self.num_eval_train = config.num_eval_train
        self.num_eval_test= config.num_eval_test
        self.is_meta = config.is_meta
        self.max_head_and_tail = None
    def update_args(self):
        if self.load_model:
            args = get_model_attribute(
                'saved_args', self.load_model_path, self.load_device)
            args.device = self.load_device
            args.load_model = True
            args.load_model_path = self.load_model_path
            args.epochs = self.epochs_end

            args.clean_tensorboard = False
            args.clean_temp = False

            args.produce_graphs = False
            args.produce_min_dfscodes = False
            args.produce_min_dfscode_tensors = False


            return args

        return self


#     def update_after_dataset(self):

#         # Model save and validate parameters
#         self.save_model = True
#         self.epochs_save = 100
#         self.epochs_validate = 20

#         # Time at which code is run

#         # Filenames to save intermediate and final outputs
#         self.fname = self.note + '_' + self.graph_type

#         # Calcuated at run time
#         self.current_model_save_path = self.model_save_path + \
#                                        self.fname + '_' + self.time + '/'

#         self.current_temp_path = self.temp_path + self.fname + '_' + self.time + '/'



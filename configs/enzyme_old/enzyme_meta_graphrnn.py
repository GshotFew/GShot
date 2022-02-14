run_type = 'meta_enzyme-graphrnn-finalwdmt0.1-k15a0.8_NEW'
run_type = 'meta_enzyme-graphrnn-finalwdmt0.0005-k15a0.8_NEW-maxprev40'
note='GraphRNN'
is_meta = True
graph_type = ''
time = '50'
GPU = 0
max_prev_node = 50
max_prev_node = 40

result_path = ''
DATASETS = ['ENZYMES_1','ENZYMES_2','ENZYMES_4', 'ENZYMES_5', 'ENZYMES_6', 'ENZYMES_3']
target_dataset = 'ENZYMES_3'
epochs_save = 100
epochs_end = 20000
epochs_validate = 100

train_val_dict = {"ENZYMES_1":(50,30,20),
                    "ENZYMES_2":(50,30,20),
                    "ENZYMES_3":(50,30,20),
                    "ENZYMES_4":(50,30,20),
                    "ENZYMES_5":(50,30,20),
                "ENZYMES_6":(50,30,20)
}
    # NUM_TRAIN_GRAPHS = 150
    # NUM_VAL_GRAPHS = 200
    # NUM_TEST_GRAPHS = 500
num_eval_train = 50
num_eval_test = 80
### extra variable for test graphs
USE_FOR_TRAINING = 50 ### fine tuning
alpha = 0.8
inner_steps = 15
# inner_steps = 15
# inner_steps = 7
# inner_steps = 3
# inner_steps = 50

weight_decay = 0.01
weight_decay_meta_train=0.1
weight_decay_meta_train=0.0005
#load_model_epoch=1000
load_epoch_tune = 5900   ### tell which model to meta test tune on
load_epoch_tune = 2300   ### tell which model to meta test tune on
load_epoch_eval = 2530    #### which model to evaluate


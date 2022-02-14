run_type = 'meta_enzyme-final-miss3-full-tain-k50-alpha0.6'
is_meta = True
graph_type = ''
time = '3'
GPU = 1
result_path = ''
DATASETS = ['ENZYMES_1','ENZYMES_2','ENZYMES_4', 'ENZYMES_5', 'ENZYMES_6', 'ENZYMES_3']
target_dataset = 'ENZYMES_3'
epochs_save = 50
epochs_end = 20000
epochs_validate = 50

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
inner_steps = 5
# inner_steps = 15
# inner_steps = 7
# inner_steps = 3
inner_steps = 50


weight_decay = 0.3 #ok

weight_decay = 0.000002 #curr

#load_model_epoch=1000
load_epoch_tune = 3450   ### tell which model to meta test tune on
load_epoch_tune = 2500   ### tell which model to meta test tune on
load_epoch_tune = 1900   ### tell which model to meta test tune on
load_epoch_tune = 1950   ### tell which model to meta test tune on
load_epoch_eval = 4025    #### which model to evaluate
load_epoch_eval = 2210    #### which model to evaluate ok


weight_decay = 0.000001 #good without curr
load_epoch_eval = 2289    #### good without curr also





weight_decay = 0.000001 #good without curr
load_epoch_eval = 2443    #### good without curr v good

load_epoch_eval = 3339    #### good without curr v good

load_epoch_eval = 2016    #### good without curr v good

load_epoch_eval = 2674    #### good without curr v good extremely amazing 0.000001  9.5 1.001

load_epoch_eval = 2200    #### threshod max default meta 9.18
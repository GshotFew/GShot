run_type = 'meta_enzyme'
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
num_eval_train = 50
num_eval_test = 80
### extra variable for test graphs
USE_FOR_TRAINING = 50 ### fine tuning
alpha = 0.8
inner_steps = 15

weight_decay_meta_train = 0.00001 #curr
load_epoch_tune = 50   ### tell which model to meta test tune on

weight_decay = 0.00001 #good without curr
load_epoch_eval = 80    #### good without curr v good extremely amazing 0.000001  9.5 1.001

threshold = 9.5
growing_factor = 1.001 
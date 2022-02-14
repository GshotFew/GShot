run_type = 'multi_enzyme-final-miss3'
is_meta = True
graph_type = ''
time = '3'
GPU = 1
result_path = ''
DATASETS = ['ENZYMES_1','ENZYMES_2','ENZYMES_4', 'ENZYMES_5', 'ENZYMES_6', 'ENZYMES_3']
target_dataset = 'ENZYMES_3'
epochs_save = 200
epochs_end = 15000
epochs_validate = 25

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
weight_decay = 0#0.3
#load_model_epoch=1000
load_epoch_tune = 200   ### tell which model to meta test tune on
load_epoch_eval = 440    #### which model to evaluate


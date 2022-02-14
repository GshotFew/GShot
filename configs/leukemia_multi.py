run_type = 'leukemia_multi'
graph_type = ''
time = '3'
GPU = 1
is_meta = True
result_path = ''
DATASETS = ['Yeast','Breast','Lung','Leukemia']
target_dataset = 'Leukemia'
epochs_save = 100
epochs_end = 10000
epochs_validate = 50
train_val_dict = {"Lung":(5000,5000,5000),
                  "Yeast":(5000,5000,5000),
                  "Breast":(5000,5000,5000),
                  "Leukemia":(500,500,900),
}

num_eval_train = 500
num_eval_test = 1000
### extra variable for test graphs
USE_FOR_TRAINING = 500 ### fine tuning
weight_decay = 0.00001#0.0005


#ignored for multi
alpha = 0.8
inner_steps = 15

# fill this using lowest validation loss model
load_epoch_tune = 800   ### tell which model to meta test tune on
load_epoch_eval = 3000    #### which model to evaluate


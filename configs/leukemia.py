run_type = 'leukemia_meta'
graph_type = ''
time = '3'
GPU = 2
is_meta = True
result_path = ''
DATASETS = ['Yeast','Breast','Lung','Leukemia']
target_dataset = 'Leukemia'
epochs_save = 100
epochs_end = 20000
epochs_validate = 100
train_val_dict = {"Lung":(5000,5000,5000),
                  "Yeast":(5000,5000,5000),
                  "Breast":(5000,5000,5000),
                  "Leukemia":(500,500,900),
}

num_eval_train = 500 #default
num_eval_test = 1000
### extra variable for test graphs
USE_FOR_TRAINING = 500
alpha = 0.8
inner_steps = 15
weight_decay_meta_train=0.00001
#load_model_epoch=1000

# fill this using lowest validation loss model
load_epoch_tune = 10000   ### tell which model to meta test tune on
load_epoch_eval = 10064    #### which model to evaluate #curr
weight_decay = 0.00001 #cur

threshold = 1.5
growing_factor= 1.001
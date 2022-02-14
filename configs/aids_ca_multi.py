run_type = 'multi_aids-ca'
graph_type = ''
time = '3'
GPU = 2
is_meta = True
result_path = ''
DATASETS = ['Yeast','Breast','Lung', 'AIDS-ca']
target_dataset = 'AIDS-ca'
epochs_save = 100
epochs_end = 10000
epochs_validate = 20
train_val_dict = {"Lung":(5000,5000,5000),
                  "Yeast":(5000,5000,5000),
                  "Breast":(5000,5000,5000),
                  "Leukemia":(500,500,500),
                  "AIDS-ca":(150,70,108)
}
num_eval_train = 150
num_eval_test = 220
### extra variable for test graphs
USE_FOR_TRAINING = 150 ### fine tuning
# alpha and inner are ignored in multi
alpha = 0.8
inner_steps = 15
weight_decay = 0.00001
#load_model_epoch=1000

# fill this using lowest validation loss model
load_epoch_tune = 2000   ### tell which model to meta test tune on
load_epoch_eval = 2440    #### which model to evaluate
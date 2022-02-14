run_type = 'AIDS-ca_single-grnn'
note='GraphRNN'
is_meta = False
graph_type = ''
time = '2'
GPU = 3
result_path = ''
DATASETS = ['AIDS-ca']
target_dataset = 'AIDS-ca'

epochs_save = 5
epochs_end = 100
epochs_validate = 5
train_val_dict = {"Lung":(5000,5000,5000),
                  "Yeast":(5000,5000,5000),
                  "Breast":(5000,5000,5000),
                  "Leukemia":(500,500,500),
                  "AIDS-ca":(150,70,108)
}
num_eval_train = 150
num_eval_test = 220
### extra variable for test graphs
load_epoch_eval = 25    #### which model to evaluate  ### can also be set after training finishes

fine_tune = False
USE_FOR_TRAINING = None
alpha = None
inner_steps = None
weight_decay = None
load_model_epoch=None
load_epoch_tune = None


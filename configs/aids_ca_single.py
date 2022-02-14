run_type = 'AIDS-ca_single'
is_meta = False
graph_type = ''
time = '1'
GPU = 3
result_path = ''
DATASETS = ['AIDS-ca']
target_dataset = 'AIDS-ca'

epochs_save = 100
epochs_end = 10000
epochs_validate = 20
train_val_dict = {
                  "AIDS-ca":(150,70,108)
}
num_eval_train = 150
num_eval_test = 220
### extra variable for test graphs
load_epoch_eval = 300    #### which model to evaluate  ### can also be set after training finishes



## Not relevant to single run but needs to be defined otherwise error will come
fine_tune = False
USE_FOR_TRAINING = None
alpha = None
inner_steps = None
weight_decay = None
load_model_epoch=None
load_epoch_tune = None


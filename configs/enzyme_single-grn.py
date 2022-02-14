run_type = 'enzyme_3-single-grnn'
note='GraphRNN'
is_meta = False
graph_type = ''
time = '1'
GPU = 3
result_path = ''
DATASETS = ['ENZYMES_3']
target_dataset = 'ENZYMES_3'

epochs_save = 300
epochs_end = 30000
epochs_validate = 50
train_val_dict = {"ENZYMES_3":(50,30,20),

}
num_eval_train = 50
num_eval_test = 80
### extra variable for test graphs
load_epoch_eval = 3600    #### which model to evaluate  ### can also be set after training finishes

## Not relevant to single run but needs to be defined otherwise error will come
fine_tune = False
USE_FOR_TRAINING = None
alpha = None
inner_steps = None
weight_decay = None
load_model_epoch=None
load_epoch_tune = None


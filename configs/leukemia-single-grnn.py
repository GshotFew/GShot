run_type = 'Leukemia-single-graphrnn'
note='GraphRNN'
is_meta = False
graph_type = ''
time = '1'
GPU = 3
result_path = ''
DATASETS = ['Leukemia']
target_dataset = 'Leukemia'

epochs_save = 300
epochs_end = 30000
epochs_validate = 20
train_val_dict = {
                  "Leukemia":(500,500,900),
                 
}
num_eval_test = 1000   ### it should be index from where test graph needs to be taken
### extra variable for test graphs
load_epoch_eval = 1800    #### which model to evaluate  ### can also be set after training finishes

## Not relevant to single run but needs to be defined otherwise error will come
num_eval_train = 500
fine_tune = False
USE_FOR_TRAINING = None
alpha = None
inner_steps = None
weight_decay = None
load_model_epoch=None
load_epoch_tune = None


run_type = 'meta_springs-5x5-456'

graph_type = ''
time = '1'
GPU = 2
is_meta = True
result_path = ''
DATASETS = ['springs_n_balls_4','springs_n_balls_6', 'springs_n_balls_5']
target_dataset = 'springs_n_balls_5'
epochs_save = 20
epochs_end = 7000
epochs_validate = 100

train_val_dict = {
                  "springs_n_balls_4":(500,500,500),
                  "springs_n_balls_6":(500,500,500),
                  "springs_n_balls_5":(500,500,500)

}

weight_decay_meta_train=0.00001

num_eval_train = 500
num_eval_test = 1000
### extra variable for test graphs
USE_FOR_TRAINING = 500
alpha = 0.8
inner_steps = 15
weight_decay = 0.00001
load_epoch_tune = 1700
load_epoch_eval = 3900


threshold = 2.5
growing_factor= 1.1
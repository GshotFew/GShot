run_type = 'multi_springs-5x5-456'
graph_type = ''
time = '2'
GPU = 1
is_meta = True
result_path = ''
DATASETS = ['springs_n_balls_4','springs_n_balls_6', 'springs_n_balls_5']
target_dataset = 'springs_n_balls_5'
epochs_save = 10
epochs_end = 7000
epochs_validate = 10


train_val_dict = {
                  "springs_n_balls_4":(500,500,500),
                  "springs_n_balls_6":(500,500,500),
                  "springs_n_balls_5":(500,500,500)

}

num_eval_train = 500
num_eval_test = 1000

### extra variable for test graphs
USE_FOR_TRAINING = 500
# alpha and inner are ignored in multi
alpha = 0.85
inner_steps = 15
weight_decay = 0.00001

# fill this using lowest validation loss model
load_epoch_tune = 460
load_epoch_eval = 7740





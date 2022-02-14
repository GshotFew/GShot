import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter

from utils import save_model, load_model, get_model_attribute
from graphgen.train import evaluate_loss as eval_loss_dfscode_rnn
from baselines.graph_rnn.train import evaluate_loss as eval_loss_graph_rnn
import torch, numpy as np
import os
import torch.nn.functional as F
import torch.nn as nn
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.seed_all()


class SPLLoss(nn.BCELoss):
    def __init__(self, *args, n_samples=0,other_args=None, **kwargs):
        super(SPLLoss, self).__init__(*args, **kwargs)

        print(args)
        self.threshold = other_args.threshold
        self.growing_factor = other_args.growing_factor
 
        print('self.threshold ', self.threshold)
        print(' self.growing_factor ', self.growing_factor)

        #self.v = torch.zeros(n_samples).int()

    def forward(self, input, target, x_len) :
        super_loss =nn.functional.binary_cross_entropy(
                input,target, reduction='none')#, weight=weight)
        
        super_loss = torch.sum(super_loss, dim=[1, 2]) / (x_len.float() + 1)

        v = self.spl_loss(super_loss)
        
        return (super_loss * v).mean()

    def increase_threshold(self):
        self.threshold *= self.growing_factor
       

    def spl_loss(self, super_loss):
        v = super_loss < self.threshold
        return v.int()

def evaluate_loss(args, model, data, feature_map, criterion_cur=None):
    if args.note == 'GraphRNN':
        loss = eval_loss_graph_rnn(args, model, data, feature_map)
    elif args.note == 'DFScodeRNN':
        loss = eval_loss_dfscode_rnn(args, model, data, feature_map, criterion_cur)
    elif args.note == 'DGMG':
        loss = eval_loss_dgmg(model, data)

    return loss


def train_epoch(
        epoch, args, model, dataloader_train, optimizer,
        scheduler, feature_map, summary_writer=None):
    # Set training mode for modules
    for _, net in model.items():
        net.train()

    batch_count = len(dataloader_train)
    total_loss = 0.0
    for batch_id, data in enumerate(dataloader_train):
        for _, net in model.items():
            net.zero_grad()

        loss = evaluate_loss(args, model, data, feature_map)

        loss.backward()
        total_loss += loss.data.item()

        # Clipping gradients
        if args.gradient_clipping:
            for _, net in model.items():
                clip_grad_value_(net.parameters(), 1.0)

        # Update params of rnn and mlp
        for _, opt in optimizer.items():
            opt.step()

        for _, sched in scheduler.items():
            sched.step()

        if args.log_tensorboard:
            summary_writer.add_scalar('{} {} Loss/train batch'.format(
                args.note, args.graph_type), loss, batch_id + batch_count * epoch)

    return total_loss / batch_count


def get_weights_copy(model, add_string_term,args):   
    #inconsitency : either put timestamp or use namestamp
    if not os.path.isdir(args.current_model_save_path):
        os.makedirs(args.current_model_save_path)
    weights_path = args.current_model_save_path+'/weights_temp-'+add_string_term
    torch.save(model.state_dict(), weights_path)
    return torch.load(weights_path)

def train_epoch_meta(
        epoch, args, model, dict_dataset_all, optimizer,
        scheduler, summary_writer=None, dict_valid_loader=None):
    # Set training mode for modules

    #print("epoch ", epoch)

    for _, net in model.items():
        net.train()

  
    total_val_loss = 0
    alpha =  args.alpha 
    inner_steps = args.inner_steps

    for dataset, dict_dataset_cur in dict_dataset_all.items():
        dataloader_train= dict_dataset_all[dataset]['dataloader_train']
        dataloader_validate = dict_dataset_all[dataset]['dataloader_validate']
        feature_map = dict_dataset_all[dataset]['feature_map']


        train_or_not = dict_dataset_cur['training']

        if train_or_not == False:
            #print("not training tuning on dataset ", dataset)
            continue

        batch_count = len(dataloader_train)
        total_loss = 0.0

        state_dict_original = {}
        for model_keys in model.keys():
            state_dict_original[model_keys] = get_weights_copy(model[model_keys], str(alpha)+str(inner_steps)+str(args.graph_type),args)


        num_inner_steps_done = 0

        for _, net in model.items():
            net.train()


        for batch_id, data in enumerate(dataloader_train):

            if (num_inner_steps_done >= inner_steps):
                break

            num_inner_steps_done += 1

            for _, net in model.items():
                net.zero_grad()

            loss = evaluate_loss(args, model, data, feature_map)
            # print("loss ", loss.item())

            loss.backward()

            total_loss += loss.data.item()

            # Clipping gradients
            if args.gradient_clipping:
                for _, net in model.items():
                    clip_grad_value_(net.parameters(), 1.0)

            # Update params of rnn and mlp
            for _, opt in optimizer.items():
                opt.step()

            if(args.schedule):
                # print('schedule ',args.schedule)
                for _, sched in scheduler.items():
                    sched.step()

            if args.log_tensorboard:
                summary_writer.add_scalar('{} {} Loss/train batch'.format(
                    args.note, args.graph_type), loss, batch_id + batch_count * epoch)


        #validate cur task
        if epoch % args.epochs_validate == 0:
           # print("Now validating dataset ", dataset)
            # fine_tuning_for_dataset_loss = train_individual(
            #     epoch, args, model, dict_train_loader[dataset], optimizer, scheduler, feature_map, writer)

            loss_validate = test_data(
                args, model, dataloader_validate, feature_map)
            #print('Epoch: {}/{}, validation loss: {:.6f}'.format(
             #   epoch, args.epochs, loss_validate))

            total_val_loss += loss_validate

#            print(" total_val_loss now is ", total_val_loss)


        candidate_weights = model

        for items_cur_weights in state_dict_original.keys():
            state_dict_item_original = state_dict_original[items_cur_weights]#.state_dict()
            state_dict_item_candidate = candidate_weights[items_cur_weights].state_dict()


            state_dict_updated = {candidate: (state_dict_item_original[candidate] + alpha *
                                                              (state_dict_item_candidate[candidate] - state_dict_item_original[candidate]))
                                                  for candidate in state_dict_item_original}

            model[items_cur_weights].load_state_dict(state_dict_updated)

            
    if ( epoch % args.epochs_validate == 0):
        print(' Epoch {}/{} Validation done with total_val_loss {} '.format(epoch,args.epochs, total_val_loss))


    return total_loss / batch_count



def train_epoch_multi(
        epoch, args, model, dict_dataset_all, optimizer,
        scheduler, summary_writer=None, dict_valid_loader=None):

    print("epoch ", epoch)

    for loop_multi in range(0,50):


        for _, net in model.items():
            net.train()

     
        total_val_loss = 0

        for _, net in model.items():
            net.zero_grad()

        for _, net in model.items():
            net.train()

        for dataset, dict_dataset_cur in dict_dataset_all.items():
            # print(" dataset in train epoch ", dataset)
            dataloader_train= dict_dataset_all[dataset]['dataloader_train']
            dataloader_validate = dict_dataset_all[dataset]['dataloader_validate']
            feature_map = dict_dataset_all[dataset]['feature_map']

            train_or_not = dict_dataset_cur['training']

            if train_or_not == False:
                # print("not training tuning on dataset ", dataset)
                continue

            batch_count = len(dataloader_train)
            total_loss = 0.0

          
            for batch_id, data in enumerate(dataloader_train):
                loss = evaluate_loss(args, model, data, feature_map)

                loss.backward()
                total_loss += loss.data.item()

                break

        # Clipping gradients
        if args.gradient_clipping:
            for _, net in model.items():
                clip_grad_value_(net.parameters(), 1.0)

        # Update params of rnn and mlp
        for _, opt in optimizer.items():
            opt.step()

        for _, sched in scheduler.items():
            sched.step()

        if args.log_tensorboard:
            summary_writer.add_scalar('{} {} Loss/train batch'.format(
                args.note, args.graph_type), loss, batch_id + batch_count * epoch)


    #validate cur task
    if epoch % args.epochs_validate == 0:
            total_val_loss=0
           
            for dataset, dict_dataset_cur in dict_dataset_all.items():
                train_or_not = dict_dataset_cur['training']

                if train_or_not == False:
                    #print("not validating  on dataset ", dataset)
                    continue

                dataloader_validate = dict_dataset_all[dataset]['dataloader_validate']
                feature_map = dict_dataset_all[dataset]['feature_map']
                loss_validate = test_data(
                    args, model, dataloader_validate, feature_map)

                total_val_loss += loss_validate

       

            
    if ( epoch % args.epochs_validate == 0):
        print(' Epoch {}/{} Validation done with total_val_loss {} '.format(epoch,args.epochs, total_val_loss))

    return total_loss / batch_count



def test_data(args, model, dataloader, feature_map, criterion_cur=None, ):
    for _, net in model.items():
        net.eval()

    batch_count = len(dataloader)
    with torch.no_grad():
        total_loss = 0.0
        for _, data in enumerate(dataloader):
            loss = evaluate_loss(args, model, data, feature_map, criterion_cur)
            total_loss += loss.data.item()

    return total_loss / batch_count


def train(args, dict_dataset_all, model): #"#", feature_map, dataloader_validate=None):
    # initialize optimizer
    optimizer = {}
    for name, net in model.items():
        optimizer['optimizer_' + name] = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
            weight_decay=5e-5)

    scheduler = {}
    for name, net in model.items():
        scheduler['scheduler_' + name] = MultiStepLR(
            optimizer['optimizer_' + name], milestones=args.milestones,
            gamma=args.gamma)

    if args.load_model:
        load_model(args.load_model_path, args.device,
                   model, optimizer, scheduler)
        print('Model loaded')

        epoch = get_model_attribute('epoch', args.load_model_path, args.device)
    else:
        epoch = 0

    if args.log_tensorboard:
        writer = SummaryWriter(
            log_dir=args.tensorboard_path + args.fname + ' ' + args.time, flush_secs=5)
    else:
        writer = None

    while epoch < args.epochs:

        if(len(dict_dataset_all)==1):
            # print("only 1 element in dict_dataset_all.  Normal training")
            dataloader_train = dict_dataset_all[list(dict_dataset_all.keys())[0]]['dataloader_train']
            dataloader_validate = dict_dataset_all[list(dict_dataset_all.keys())[0]]['dataloader_validate']
            feature_map = dict_dataset_all[list(dict_dataset_all.keys())[0]]['feature_map']

            loss = train_epoch(
                epoch, args, model, dataloader_train, optimizer, scheduler, feature_map, writer)
            epoch += 1

            # logging
            if args.log_tensorboard:
                writer.add_scalar('{} {} Loss/train'.format(
                    args.note, args.graph_type), loss, epoch)
            else:
                print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))

            # save model checkpoint
            if args.save_model and epoch != 0 and epoch % args.epochs_save == 0:
                save_model(
                    epoch, args, model, optimizer, scheduler, feature_map=feature_map)
                print(
                    'Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))

            if dataloader_validate is not None and epoch % args.epochs_validate == 0:
                loss_validate = test_data(
                    args, model, dataloader_validate, feature_map)
            
                print('Epoch: {}/{}, validation loss: {:.6f}'.format(
                        epoch, args.epochs, loss_validate))
    #
    save_model(epoch, args, model, optimizer,
               scheduler, feature_map=feature_map)
    print('Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))


def train_meta(args, dict_dataset_all, model):
    # initialize optimizer
    optimizer = {}
    for name, net in model.items():
        optimizer['optimizer_' + name] = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
            
            weight_decay=args.weight_decay_meta_train)
            # weight_decay=5e-5)
    print(' weight_decay_meta_train ', args.weight_decay_meta_train)

    scheduler = {}
    for name, net in model.items():
        scheduler['scheduler_' + name] = MultiStepLR(
            optimizer['optimizer_' + name], milestones=args.milestones,
            gamma=args.gamma)

    if args.load_model:
        load_model(args.load_model_path, args.device,
                   model, optimizer, scheduler)
        print('Model loaded')

        epoch = get_model_attribute('epoch', args.load_model_path, args.device)
    else:
        epoch = 0

    if args.log_tensorboard:
        writer = SummaryWriter(
            log_dir=args.tensorboard_path + args.fname + ' ' + args.time, flush_secs=5)
    else:
        writer = None

    print('args.epochs ', args.epochs)
    while epoch < args.epochs:

            # print("feature map")
            feature_map = dict_dataset_all[list(dict_dataset_all.keys())[0]]['feature_map']

            loss = train_epoch_meta(
                epoch, args, model, dict_dataset_all, optimizer, scheduler, writer)
            epoch += 1

            # logging
            if args.log_tensorboard:
                writer.add_scalar('{} {} Loss/train'.format(
                    args.note, args.graph_type), loss, epoch)
            else:
                print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))

            # save model checkpoint
            if args.save_model and epoch != 0 and epoch % args.epochs_save == 0:
                save_model(
                    epoch, args, model, optimizer, scheduler, feature_map=feature_map)
                print(
                    'Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))

          
    print('Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))


def fine_tune_target(
        epoch, args, model, criterion_cur, dataloader_train, dataloader_validate, feature_map, optimizer,
        scheduler, summary_writer=None, dict_valid_loader=None):
    # Set training mode for modules

    for _, net in model.items():
        net.train()

 
    total_val_loss = 0
    # alpha =0.8
    alpha =1
    # print("alpha ", alpha)
    inner_steps = 1
   

    batch_count = len(dataloader_train)
    total_loss = 0.0

    num_inner_steps_done = 0
    # while True:

    for _, net in model.items():
        net.train()


    for batch_id, data in enumerate(dataloader_train):
        # print("batch_id ", batch_id)

        if (num_inner_steps_done >= inner_steps):
            break

        num_inner_steps_done += 1

        for _, net in model.items():
            net.zero_grad()


        loss = evaluate_loss(args, model, data, feature_map, criterion_cur)

  

        loss.backward()

        total_loss += loss.data.item()

        # Clipping gradients
        if args.gradient_clipping:
            for _, net in model.items():
                clip_grad_value_(net.parameters(), 1.0)

        # Update params of rnn and mlp
        for _, opt in optimizer.items():
            opt.step()

        if(args.schedule):
            # print('schedule ',args.schedule)
            for _, sched in scheduler.items():
                sched.step()

        if args.log_tensorboard:
            summary_writer.add_scalar('{} {} Loss/train batch'.format(
                args.note, args.graph_type), loss, batch_id + batch_count * epoch)


    #validate cur task
    if epoch % args.epochs_validate == 0:
        print("Now validating dataset ")
       
        loss_validate = test_data(
            args, model, dataloader_validate, feature_map)
        print('Epoch: {}/{}, validation loss: {:.6f}'.format(
            epoch, args.epochs, loss_validate))

        total_val_loss += loss_validate

        print(" total_val_loss now is ", total_val_loss)


    return total_loss / batch_count



def fine_tune(args, dict_dataset_all, target_dataset_name, model ):
    # initialize optimizer

    dataloader_train = dict_dataset_all[target_dataset_name]['dataloader_train']
    dataloader_validate = dict_dataset_all[target_dataset_name]['dataloader_validate']

    optimizer = {}
    weight_decay = args.weight_decay
    print('weight_decay ',weight_decay)
    for name, net in model.items():
        optimizer['optimizer_' + name] = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
            # filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr*0.1,
            # weight_decay=5e-5)
            weight_decay=weight_decay)
    print(' lr ',args.lr)
    scheduler = {}
    for name, net in model.items():
        scheduler['scheduler_' + name] = MultiStepLR(
            optimizer['optimizer_' + name], milestones=args.milestones,
            gamma=args.gamma)

    if args.load_model:
        print('args.load_model_path ', args.load_model_path)


        load_model(args.load_model_path, args.device,
                   model, optimizer, scheduler)


        for name, net in model.items():
            optimizer['optimizer_' + name].param_groups[0]['weight_decay']  =    weight_decay

            print(' updated wdecay', weight_decay)
      

        print('Model loaded')

        epoch = get_model_attribute('epoch', args.load_model_path, args.device)

    else:
        epoch = 0
        print(' model not loaded ', exit)
        exit()

    if args.log_tensorboard:
        writer = SummaryWriter(
            log_dir=args.tensorboard_path + args.fname + ' ' + args.time, flush_secs=5)
    else:
        writer = None


    criterion_cur = SPLLoss(n_samples=args.batch_size, other_args=args)

    while epoch < args.epochs:

            # print("feature map")
            feature_map = dict_dataset_all[list(dict_dataset_all.keys())[0]]['feature_map']

            loss = fine_tune_target(
                epoch, args, model, criterion_cur, dataloader_train, dataloader_validate, feature_map,  optimizer, scheduler, writer)
            epoch += 1

            # logging
            if args.log_tensorboard:
                writer.add_scalar('{} {} Loss/train'.format(
                    args.note, args.graph_type), loss, epoch)
            else:
                print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))

            # save model checkpoint
            if args.save_model and epoch != 0 and epoch % args.epochs_save == 0:
                save_model(
                    epoch, args, model, optimizer, scheduler, feature_map=feature_map, fine_tuned_name=target_dataset_name)
                print(
                    'Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))

            criterion_cur.increase_threshold()

    print('Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))



def train_multi(args, dict_dataset_all, model):
    # initialize optimizer
    optimizer = {}
    for name, net in model.items():
        optimizer['optimizer_' + name] = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
            weight_decay=5e-5)

    scheduler = {}
    for name, net in model.items():
        scheduler['scheduler_' + name] = MultiStepLR(
            optimizer['optimizer_' + name], milestones=args.milestones,
            gamma=args.gamma)

    if args.load_model:
        load_model(args.load_model_path, args.device,
                   model, optimizer, scheduler)
        print('Model loaded')

        epoch = get_model_attribute('epoch', args.load_model_path, args.device)
    else:
        epoch = 0

    if args.log_tensorboard:
        writer = SummaryWriter(
            log_dir=args.tensorboard_path + args.fname + ' ' + args.time, flush_secs=5)
    else:
        writer = None

    while epoch < args.epochs:

            print("feature map")
            feature_map = dict_dataset_all[list(dict_dataset_all.keys())[0]]['feature_map']

            loss = train_epoch_multi(
                epoch, args, model, dict_dataset_all, optimizer, scheduler, writer)
            epoch += 1

            # logging
            if args.log_tensorboard:
                writer.add_scalar('{} {} Loss/train'.format(
                    args.note, args.graph_type), loss, epoch)
            else:
                print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))

            # save model checkpoint
            if args.save_model and epoch != 0 and epoch % args.epochs_save == 0:
                save_model(
                    epoch, args, model, optimizer, scheduler, feature_map=feature_map)
                print(
                    'Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))

  
    print('Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))

from baselines.graph_rnn.model import create_model as create_model_graph_rnn
from graphgen.model import create_model as create_model_graphgen
import torch, numpy as np
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.seed_all()

def create_model(args, feature_map):




    if args.note == 'GraphRNN':
        model = create_model_graph_rnn(args, feature_map)

    elif args.note == 'DFScodeRNN':
        model = create_model_graphgen(args, feature_map)


    return model

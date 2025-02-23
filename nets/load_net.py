"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.gcn_net import GCNNet
from nets.mlp_net import MLPNet
from nets.brain_net_cnn import BrainNetCNN
from nets.brainpromptg_net import BrainPromptGNet
from nets.brainpromptc_net import BrainPromptCNet


def MLP(net_params, trainset):
    return MLPNet(net_params)

def GCN(net_params, trainset):
    return GCNNet(net_params)


def BrainCNN(net_params, trainset):
    return BrainNetCNN(net_params)


def BrainPromptG(net_params, trainset):
    return BrainPromptGNet(net_params)


def BrainPromptC(net_params, trainset):
    return BrainPromptCNet(net_params)


def gnn_model(MODEL_NAME, net_params, trainset):
    models = {
        'GCN': GCN,
        'MLP': MLP,
        'BrainNetCNN': BrainCNN,
        "BrainPromptG": BrainPromptG,
        "BrainPromptC": BrainPromptC,
    }
    model = models[MODEL_NAME](net_params, trainset)
    model.name = MODEL_NAME
        
    return model

"""
    File to load dataset based on user control from main file
"""
from data.BrainNet import name2path, BrainDataset_llm


def LoadData_llm(DATASET_NAME, threshold=0, edge_ratio=0, node_feat_transform='original'):
    """
        This function is called in the main.py file
        returns:
        ; dataset object
    """

    if DATASET_NAME in name2path.keys():
        return BrainDataset_llm(DATASET_NAME, threshold=threshold, edge_ratio=edge_ratio,
                            node_feat_transform=node_feat_transform)

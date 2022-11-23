import pandas as pd


class Node:
    def __init__(self, df=pd.DataFrame(), feature='', threshold_value=0.0, is_leaf=True, child_nodes=[], target_value=None, depth=0):
        self.df = df
        self.feature = feature
        self.threshold_value = threshold_value
        self.is_leaf = is_leaf
        self.child_nodes = child_nodes
        self.target_value = target_value
        self.depth = depth
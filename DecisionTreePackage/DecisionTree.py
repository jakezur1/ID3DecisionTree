import math
import pandas as pd
from Node import Node


class DecisionTree:
    root = None
    def __init__(self, df, features, target):
        self.df = df
        self.features = features
        self.target = target
        self.root = Node(self.df, feature = '', threshold_value=0.0, is_leaf=True, child_nodes=[])

    def calc_entropy(self, split_df):
        target_occurences = split_df[self.target].value_counts().to_list()
        entropy = 0.0
        for value in target_occurences:
            probability = value/len(split_df)
            entropy-=probability*math.log2(probability)
        return entropy
    
    def fit_model(self, root=None, iteration=0):
        if iteration == 0:
            root = self.root
        df = root.df
        root_entropy = self.calc_entropy(df)
        information_gain = 0.0
        left_node = Node()
        right_node = Node()
        for feature in self.features:
            feature_list = df[feature].to_list()
            df_left = pd.DataFrame()
            df_right = pd.DataFrame()
            min_val = min(feature_list)
            max_val = max(feature_list)
            step_size = (max_val-min_val)/len(df)
            iterable_list = []
            num_to_add = min_val
            while num_to_add < max_val-step_size:
                num_to_add+=step_size
                iterable_list.append(num_to_add)
            for threshold_val in iterable_list:
                df_left = df[df[feature] < threshold_val]
                df_right = df[df[feature] >= threshold_val]
                left_entropy = self.calc_entropy(df_left)
                right_entropy = self.calc_entropy(df_right)
                curr_info_gain = abs(root_entropy - (len(df_left)/len(df))*left_entropy - (len(df_right)/len(df))*right_entropy)
                if curr_info_gain > information_gain:
                    information_gain = curr_info_gain
                    left_node.df = df_left
                    left_node.target_value = df_left[self.target].value_counts().idxmax()
                    right_node.df = df_right
                    right_node.target_value = df_right[self.target].value_counts().idxmax()
                    root.feature = feature
                    root.child_nodes = [left_node, right_node]
                    root.threshold_value = threshold_val
                    root.is_leaf = False
                    root.depth=iteration
        if root.is_leaf == False:
            for child in root.child_nodes:
                self.fit_model(child, iteration=iteration+1)

    def traverse_tree(self, df, root=None, iteration=0):
        predictions = []
        prediction_df = df.copy()
        for row in range(len(df)):
            prediction = self.traverse_branch(df=df, root=None, iteration=0, row=row)
            predictions.append(prediction)
        prediction_df['prediction'] = predictions
        return prediction_df
    
    def traverse_branch(self, df, root=None, iteration=0, row=0):
        if iteration == 0:
            root=self.root
        if root.is_leaf:
            return root.target_value
        else:
            if df[root.feature].to_list()[row] < root.threshold_value:
                return self.traverse_branch(df=df, root=root.child_nodes[0], iteration=iteration+1, row=row)
            else:
                return self.traverse_branch(df=df, root=root.child_nodes[1], iteration=iteration+1, row=row)

    def model_accuracy(self, df):
        model_df = self.traverse_tree(df=df)
        target_list = model_df[self.target].to_list()
        prediction_list = model_df['prediction'].to_list()
        num_correct = 0
        for i, value in enumerate(target_list):
            if value == prediction_list[i]:
                num_correct+=1
        accuracy_score = round(100*num_correct/len(target_list), 2)
        return accuracy_score

    # needs work...
    def visualize_model(self, root=None, iteration=0):
        if iteration == 0:
            root = self.root
        if root.is_leaf == False:
            print(4 * root.depth * '-' + "Feature: " + str(root.feature))
            print(4 * root.depth * ' ' + "Threshold value: " + str(root.threshold_value))
            print(4 * root.depth * ' ' + "Target value: " + str(root.target_value))
            self.visualize_model(root=root.child_nodes[0], iteration=iteration+1)
            self.visualize_model(root=root.child_nodes[1], iteration=iteration+1)
    
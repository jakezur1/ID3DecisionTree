import sys
sys.path.insert(1, '/Users/jakezur/Documents/boredom-projects/DecisionTreeFromScratch/DecisionTreePackage')
from DecisionTree import DecisionTree
import pandas as pd
import random


df = pd.read_csv("/Users/jakezur/Documents/boredom-projects/DecisionTreeModel/Iris/iris.csv")

df_permutated = df.sample(frac=1)

train_size = 0.8
train_end = int(len(df_permutated)*train_size)

df_train = df_permutated[:train_end]

df_test = df_permutated[train_end:]

random_row = random.randint(0, len(df_test))

dt = DecisionTree(df=df_train, features=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'], target='variety')

dt.fit_model()
predictions = dt.traverse_tree(df=df_test)
print(predictions)
print(' ')

accuracy_score = dt.model_accuracy(df_test)
print('This model is ' + str(accuracy_score) + '% accuracte.')
print(' ')
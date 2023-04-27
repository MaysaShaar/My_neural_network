import numpy as np
import pandas as pd
from networkClass import My_network

data = np.random.rand(10, 2, 1)
y = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 0])

 
# number of hibben layers + the output layer
layers = 4
layers_nodes = [2, 3, 5, 2, 1] # inedx 0 is the input layer, -1 index is the output layer
activations = ["tanh", "sigmoid", "tanh", "sigmoid"]

iterations = 1000
alpha = 0.1

Mynetwork = My_network(data, y, layers, layers_nodes, activations, iterations, alpha)
ypred, error =  Mynetwork.fit()

    
df = pd.DataFrame({"x1":data[:, 0, 0], "x2":data[:, 1, 0], "y":y, "predect":ypred })
print(df)

import torch.nn as nn

class Model(nn.Module):
    def __init__(self, embedding_size):
        super(Model, self).__init__()             
        self.linear0 = nn.Linear(3, embedding_size)
        self.linear1 = nn.Linear(embedding_size, 2)
        self.linear_layers = [self.linear0, self.linear1]
        self.relu = nn.ReLU()
        
    def forward(self, x):   
        x = self.linear0(x)
        x = self.relu(x)  
        x = self.linear1(x)
        return x

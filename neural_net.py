import torch
import torch.nn as nn
import torch.nn.functional as F

input_nodes = 7
hidden_nodes_1 = 9
hidden_nodes_2 = 15
output_nodes = 3

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_nodes, hidden_nodes_1)
        self.fc2 = nn.Linear(hidden_nodes_1, hidden_nodes_2)
        self.fc3 = nn.Linear(hidden_nodes_2, output_nodes)  
        
    def fill_weights(self, weights):

        fc1_end = input_nodes * hidden_nodes_1
        fc2_end = fc1_end + hidden_nodes_1 * hidden_nodes_2
        fc3_end = fc2_end + hidden_nodes_2 * output_nodes
        
        self.fc1.weight.data = torch.tensor(weights[:fc1_end].reshape(hidden_nodes_1, input_nodes), dtype=torch.float)
        
        self.fc2.weight.data = torch.tensor(weights[fc1_end:fc2_end].reshape(hidden_nodes_2, hidden_nodes_1), dtype=torch.float)
        
        self.fc3.weight.data = torch.tensor(weights[fc2_end:fc3_end].reshape(output_nodes, hidden_nodes_2), dtype=torch.float)
        
    def update_individual_weights(self, index, weight):
        fc1_end = input_nodes * hidden_nodes_1  
        fc2_end = fc1_end + hidden_nodes_1 * hidden_nodes_2

        if index < fc1_end:
            flat_index = index
            self.fc1.weight.data.view(-1)[flat_index] = weight
        elif index < fc2_end:
            flat_index = index - fc1_end
            self.fc2.weight.data.view(-1)[flat_index] = weight
        else:
            flat_index = index - fc2_end
            self.fc3.weight.data.view(-1)[flat_index] = weight

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  
        x = torch.tanh(self.fc2(x))  
        x = self.fc3(x)              
        return F.softmax(x, dim=1)   

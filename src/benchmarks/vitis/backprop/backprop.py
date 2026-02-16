import torch
import torch_mlir

class backprop(torch.nn.Module):
    def __init__(self, input_dimension, nodes_per_layer, possible_outputs, learning_rate=0.01):
        super(backprop, self).__init__()
        
        self.input_dimension = input_dimension
        self.nodes_per_layer = nodes_per_layer
        self.possible_outputs = possible_outputs
        
        # Register learning rate as buffer (1D to avoid scalar tensors)
        self.register_buffer('learning_rate', torch.tensor([learning_rate]))
        
        # Initialize weights and biases as parameters
        # Layer 1: input -> hidden1
        self.weights1 = torch.nn.Parameter(
            torch.full((input_dimension, nodes_per_layer), 0.01)
        )
        self.biases1 = torch.nn.Parameter(torch.zeros(nodes_per_layer))
        
        # Layer 2: hidden1 -> hidden2
        self.weights2 = torch.nn.Parameter(
            torch.full((nodes_per_layer, nodes_per_layer), 0.01)
        )
        self.biases2 = torch.nn.Parameter(torch.zeros(nodes_per_layer))
        
        # Layer 3: hidden2 -> output
        self.weights3 = torch.nn.Parameter(
            torch.full((nodes_per_layer, possible_outputs), 0.01)
        )
        self.biases3 = torch.nn.Parameter(torch.zeros(possible_outputs))
    
    def sigmoid(self, x):
        """Sigmoid activation (called RELU in original but implements sigmoid)"""
        return torch.sigmoid(x)
    
    def softmax(self, x):
        """Softmax with negative values (unusual but matches C code)"""
        neg_x = torch.neg(x)
        exp_neg_x = torch.exp(neg_x)
        sum_exp = torch.sum(exp_neg_x, dim=0, keepdim=True)
        return exp_neg_x / sum_exp
    
    def forward_pass(self, input_sample, weights1=None, biases1=None, weights2=None, biases2=None, weights3=None, biases3=None):
        """Forward pass through the network"""
        if weights1 is None:
            weights1 = self.weights1
        if biases1 is None:
            biases1 = self.biases1
        if weights2 is None:
            weights2 = self.weights2
        if biases2 is None:
            biases2 = self.biases2
        if weights3 is None:
            weights3 = self.weights3
        if biases3 is None:
            biases3 = self.biases3
        # Layer 1
        z1 = torch.matmul(input_sample, weights1) + biases1
        a1 = self.sigmoid(z1)
        
        # Layer 2
        z2 = torch.matmul(a1, weights2) + biases2
        a2 = self.sigmoid(z2)
        
        # Layer 3
        z3 = torch.matmul(a2, weights3) + biases3
        a3 = self.sigmoid(z3)
        
        # Softmax output
        net_output = self.softmax(a3)
        
        return net_output, a1, a2, a3, z1, z2, z3
    
    def normalize_weights(self, weights, biases):
        """Normalize weights and biases by their L2 norm"""
        weight_norm = torch.sqrt(
            torch.sum(weights.reshape(-1) * weights.reshape(-1), dim=0, keepdim=True)
        )
        bias_norm = torch.sqrt(
            torch.sum(biases.reshape(-1) * biases.reshape(-1), dim=0, keepdim=True)
        )
        
        # Avoid division by zero
        weight_norm = torch.clamp(weight_norm, min=1e-8)
        bias_norm = torch.clamp(bias_norm, min=1e-8)
        
        normalized_weights = torch.div(weights, weight_norm)
        normalized_biases = torch.div(biases, bias_norm)
        
        return normalized_weights, normalized_biases
    
    def backprop(self, training_data, training_targets):
        """
        Backpropagation training step - named to match C function
        
        Args:
            training_data: [training_sets, input_dimension] tensor
            training_targets: [training_sets, possible_outputs] tensor
        
        Returns:
            Final weights after training (for compatibility)
        """
        training_sets = training_data.size(0)
        # Use local copies to keep the computation functional for Torch-MLIR.
        weights1 = self.weights1
        biases1 = self.biases1
        weights2 = self.weights2
        biases2 = self.biases2
        weights3 = self.weights3
        biases3 = self.biases3
        
        for i in range(training_sets):
            # Get single training sample
            input_sample = training_data[i]
            target = training_targets[i]
            
            # Forward pass
            net_output, a1, a2, a3, z1, z2, z3 = self.forward_pass(
                input_sample,
                weights1=weights1,
                biases1=biases1,
                weights2=weights2,
                biases2=biases2,
                weights3=weights3,
                biases3=biases3,
            )
            
            # Compute sigmoid derivatives (for backprop)
            da1 = torch.mul(a1, 1.0 - a1)
            da2 = torch.mul(a2, 1.0 - a2)
            da3 = torch.mul(a3, 1.0 - a3)
            
            # Output layer error: (target - output) * derivative
            output_diff = torch.mul((target - net_output), da3)
            
            # Backpropagation through layer 3
            delta_w3 = torch.matmul(a2.unsqueeze(1), output_diff.unsqueeze(0))
            oracle_a2 = torch.mul(
                torch.matmul(weights3, output_diff),
                da2
            )
            
            # Backpropagation through layer 2
            delta_w2 = torch.matmul(a1.unsqueeze(1), oracle_a2.unsqueeze(0))
            oracle_a1 = torch.mul(
                torch.matmul(weights2, oracle_a2),
                da1
            )
            
            # Backpropagation through layer 1
            delta_w1 = torch.matmul(input_sample.unsqueeze(1), oracle_a1.unsqueeze(0))
            
            # Gradient descent updates (functional, no in-place ops)
            weights1 = weights1 - torch.mul(delta_w1, self.learning_rate)
            biases1 = biases1 - torch.mul(oracle_a1, self.learning_rate)

            weights2 = weights2 - torch.mul(delta_w2, self.learning_rate)
            biases2 = biases2 - torch.mul(oracle_a2, self.learning_rate)

            weights3 = weights3 - torch.mul(delta_w3, self.learning_rate)
            biases3 = biases3 - torch.mul(output_diff, self.learning_rate)

            # Normalize weights and biases after each update
            weights1, biases1 = self.normalize_weights(weights1, biases1)
            weights2, biases2 = self.normalize_weights(weights2, biases2)
            weights3, biases3 = self.normalize_weights(weights3, biases3)
        
        return weights1, weights2, weights3
    
    def forward(self, training_data, training_targets):
        """Forward method calls backprop for PyTorch compatibility"""
        return self.backprop(training_data, training_targets)
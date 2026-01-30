import torch
import torch_mlir

class Backprop(torch.nn.Module):
    def __init__(self, input_dimension, nodes_per_layer, possible_outputs, learning_rate=0.01):
        super(Backprop, self).__init__()
        
        self.input_dimension = input_dimension
        self.nodes_per_layer = nodes_per_layer
        self.possible_outputs = possible_outputs
        
        # Register learning rate as buffer
        self.register_buffer('learning_rate', torch.tensor(learning_rate))
        
        # Initialize weights and biases as parameters
        # Layer 1: input -> hidden1
        self.weights1 = torch.nn.Parameter(
            torch.randn(input_dimension, nodes_per_layer) * 0.01
        )
        self.biases1 = torch.nn.Parameter(torch.zeros(nodes_per_layer))
        
        # Layer 2: hidden1 -> hidden2
        self.weights2 = torch.nn.Parameter(
            torch.randn(nodes_per_layer, nodes_per_layer) * 0.01
        )
        self.biases2 = torch.nn.Parameter(torch.zeros(nodes_per_layer))
        
        # Layer 3: hidden2 -> output
        self.weights3 = torch.nn.Parameter(
            torch.randn(nodes_per_layer, possible_outputs) * 0.01
        )
        self.biases3 = torch.nn.Parameter(torch.zeros(possible_outputs))
    
    def sigmoid(self, x):
        """Sigmoid activation (called RELU in original but implements sigmoid)"""
        return torch.sigmoid(x)
    
    def softmax(self, x):
        """Softmax with negative values (unusual but matches C code)"""
        neg_x = torch.neg(x)
        exp_neg_x = torch.exp(neg_x)
        sum_exp = torch.sum(exp_neg_x)
        return exp_neg_x / sum_exp
    
    def forward_pass(self, input_sample):
        """Forward pass through the network"""
        # Layer 1
        z1 = torch.matmul(input_sample, self.weights1) + self.biases1
        a1 = self.sigmoid(z1)
        
        # Layer 2
        z2 = torch.matmul(a1, self.weights2) + self.biases2
        a2 = self.sigmoid(z2)
        
        # Layer 3
        z3 = torch.matmul(a2, self.weights3) + self.biases3
        a3 = self.sigmoid(z3)
        
        # Softmax output
        net_output = self.softmax(a3)
        
        return net_output, a1, a2, a3, z1, z2, z3
    
    def normalize_weights(self, weights, biases):
        """Normalize weights and biases by their L2 norm"""
        weight_norm = torch.sqrt(torch.sum(torch.mul(weights, weights)))
        bias_norm = torch.sqrt(torch.sum(torch.mul(biases, biases)))
        
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
        
        for i in range(training_sets):
            # Get single training sample
            input_sample = training_data[i]
            target = training_targets[i]
            
            # Forward pass
            net_output, a1, a2, a3, z1, z2, z3 = self.forward_pass(input_sample)
            
            # Compute sigmoid derivatives (for backprop)
            da1 = torch.mul(a1, 1.0 - a1)
            da2 = torch.mul(a2, 1.0 - a2)
            da3 = torch.mul(a3, 1.0 - a3)
            
            # Output layer error: (target - output) * derivative
            output_diff = torch.mul((target - net_output), da3)
            
            # Backpropagation through layer 3
            delta_w3 = torch.outer(a2, output_diff)
            oracle_a2 = torch.mul(
                torch.matmul(self.weights3, output_diff),
                da2
            )
            
            # Backpropagation through layer 2
            delta_w2 = torch.outer(a1, oracle_a2)
            oracle_a1 = torch.mul(
                torch.matmul(self.weights2, oracle_a2),
                da1
            )
            
            # Backpropagation through layer 1
            delta_w1 = torch.outer(input_sample, oracle_a1)
            
            # Update weights with gradient descent
            with torch.no_grad():
                self.weights1.sub_(torch.mul(delta_w1, self.learning_rate))
                self.biases1.sub_(torch.mul(oracle_a1, self.learning_rate))
                
                self.weights2.sub_(torch.mul(delta_w2, self.learning_rate))
                self.biases2.sub_(torch.mul(oracle_a2, self.learning_rate))
                
                self.weights3.sub_(torch.mul(delta_w3, self.learning_rate))
                self.biases3.sub_(torch.mul(output_diff, self.learning_rate))
                
                # Normalize weights and biases after each update
                self.weights1.data, self.biases1.data = self.normalize_weights(
                    self.weights1.data, self.biases1.data
                )
                self.weights2.data, self.biases2.data = self.normalize_weights(
                    self.weights2.data, self.biases2.data
                )
                self.weights3.data, self.biases3.data = self.normalize_weights(
                    self.weights3.data, self.biases3.data
                )
        
        return self.weights1, self.weights2, self.weights3
    
    def forward(self, training_data, training_targets):
        """Forward method calls backprop for PyTorch compatibility"""
        return self.backprop(training_data, training_targets)
import torch
import torch_mlir

class sgd_sw(torch.nn.Module):
    def __init__(self, num_features, num_training, num_epochs, step_size):
        super(sgd_sw, self).__init__()
        self.num_features = num_features
        self.num_training = num_training
        self.num_epochs = num_epochs
        self.register_buffer('step_size', torch.tensor(step_size))
    
    def forward(self, data, label, theta):
        # data shape: [NUM_TRAINING, NUM_FEATURES]
        # label shape: [NUM_TRAINING]
        # theta shape: [NUM_FEATURES]
        
        # Reshape data from flat array to 2D if needed
        # data is NUM_FEATURES * NUM_TRAINING flattened
        data_2d = data.view(self.num_training, self.num_features)
        
        # Main loop over epochs (batch gradient descent)
        for epoch in range(self.num_epochs):
            # Vectorized forward pass for all training samples
            logits = torch.sum(data_2d * theta, dim=1)
            prob = torch.sigmoid(logits)

            # Compute gradient across all samples
            error = prob - label
            gradient = torch.sum(error.unsqueeze(1) * data_2d, dim=0)

            # Update theta: theta -= step_size * gradient
            theta = theta - self.step_size * gradient
        
        return theta
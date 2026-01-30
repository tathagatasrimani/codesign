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
        
        # Main loop over epochs
        for epoch in range(self.num_epochs):
            # Loop over each training instance
            for training_id in range(self.num_training):
                # Get current data sample
                data_sample = data_2d[training_id]  # shape: [NUM_FEATURES]
                
                # Dot product between theta and data sample
                dot = torch.dot(theta, data_sample)
                
                # Sigmoid: prob = 1.0 / (1.0 + exp(-dot))
                prob = torch.sigmoid(dot)
                
                # Compute gradient: (prob - label) * data_sample
                error = prob - label[training_id]
                gradient = error * data_sample
                
                # Update theta: theta -= step_size * gradient
                theta = theta - self.step_size * gradient
        
        return theta
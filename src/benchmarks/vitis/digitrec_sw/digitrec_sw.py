import torch
import torch_mlir

class digitrec_sw(torch.nn.Module):
    def __init__(self, k_const=3, num_classes=10, class_size=180):
        super(digitrec_sw, self).__init__()
        
        self.k_const = k_const
        self.num_classes = num_classes
        self.class_size = class_size
        
        # Register constants as buffers
        self.register_buffer('max_distance', torch.tensor(256))
    
    def hamming_distance(self, test_sample, train_sample):
        """
        Compute Hamming distance between two samples
        Uses XOR and popcount approximation
        
        Args:
            test_sample: [DIGIT_WIDTH] tensor
            train_sample: [DIGIT_WIDTH] tensor
        
        Returns:
            distance: scalar tensor
        """
        # XOR to find differing bits
        diff = torch.bitwise_xor(test_sample, train_sample)
        
        # Count number of 1s (Hamming weight) - sum of bits
        # For uint8, this is just the sum of bit positions
        # PyTorch doesn't have efficient popcount, so we approximate
        # by counting bits manually
        distance = torch.sum(diff.to(torch.int32))
        
        return distance
    
    def forward(self, training_set, test_set):
        """
        k-NN digit recognition
        
        Args:
            training_set: [NUM_TRAINING, DIGIT_WIDTH] tensor of uint8
            test_set: [NUM_TEST, DIGIT_WIDTH] tensor of uint8
        
        Returns:
            results: [NUM_TEST] tensor with predicted labels
        """
        num_test = test_set.size(0)
        num_training = training_set.size(0)
        
        results = torch.zeros(num_test, dtype=torch.long)
        
        # Process each test sample
        for t in range(num_test):
            # Initialize nearest neighbor tracking
            # distances to nearest k neighbors
            dists = torch.full((self.k_const,), 256, dtype=torch.int32)
            # labels of nearest k neighbors
            labels = torch.zeros(self.k_const, dtype=torch.long)
            
            test_sample = test_set[t]
            
            # Compare with each training sample
            for i in range(num_training):
                train_sample = training_set[i]
                
                # Compute Hamming distance
                dist = self.hamming_distance(test_sample, train_sample)
                
                # Find the maximum distance in current k neighbors
                max_dist = torch.max(dists)
                max_dist_idx = torch.argmax(dists)
                
                # If this training sample is closer, replace the farthest neighbor
                if dist < max_dist:
                    dists[max_dist_idx] = dist
                    labels[max_dist_idx] = i // self.class_size
            
            # Vote among k nearest neighbors
            votes = torch.zeros(self.num_classes, dtype=torch.long)
            for i in range(self.k_const):
                label_idx = labels[i]
                votes[label_idx] = votes[label_idx] + 1
            
            # Find label with maximum votes
            max_vote_idx = torch.argmax(votes)
            results[t] = max_vote_idx
        
        return results
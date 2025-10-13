import torch
import torch.nn as nn
import torch.nn.functional as F

# Base class for mini-models
class BaseMiniIA(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mem_usage_bytes = total_params * 4  # Assuming float32
        return {
            'params': total_params,
            'memory_kb': mem_usage_bytes / 1024
        }

# 1. IncompatibilityPropagator (GNN-like structure for graph data)
class IncompatibilityPropagator(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, output_dim=1, hidden_dim=64, num_layers=2):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.num_layers = num_layers

    def forward(self, node_features, edge_features, adj_matrix):
        # node_features: (num_nodes, node_feature_dim)
        # edge_features: (num_edges, edge_feature_dim)
        # adj_matrix: (num_nodes, num_nodes) - adjacency matrix

        h_nodes = self.node_mlp(node_features)
        h_edges = self.edge_mlp(edge_features) # Simplified, assumes edge_features are pre-mapped to nodes or aggregated

        # Simple message passing for demonstration
        for _ in range(self.num_layers):
            # Aggregate messages from neighbors
            aggregated_messages = torch.matmul(adj_matrix, h_nodes) # (num_nodes, hidden_dim)

            # Combine with self features and edge features (simplified)
            # For a real GNN, edge features would be used more explicitly in message passing
            combined = torch.cat([h_nodes, aggregated_messages, h_edges[:h_nodes.shape[0]]], dim=-1) # Simplified
            h_nodes = h_nodes + self.message_mlp(combined) # Residual connection

        return torch.sigmoid(self.output_mlp(h_nodes))

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mem_usage_bytes = total_params * 4  # Assuming float32
        return {
            'params': total_params,
            'memory_kb': mem_usage_bytes / 1024
        }

# 2. GlobalConstraintDecomposer (Seq2Seq-like for sequence generation)
class GlobalConstraintDecomposer(nn.Module):
    def __init__(self, input_dim, output_seq_len, vocab_size, hidden_dim=128, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(vocab_size, hidden_dim, num_layers, batch_first=True)
        self.output_linear = nn.Linear(hidden_dim, vocab_size)
        self.output_seq_len = output_seq_len
        self.vocab_size = vocab_size

    def forward(self, input_sequence, target_sequence=None):
        # input_sequence: (batch_size, seq_len, input_dim)
        # target_sequence: (batch_size, output_seq_len, vocab_size) - one-hot encoded

        encoder_outputs, (hidden, cell) = self.encoder(input_sequence)

        decoder_input = torch.zeros(input_sequence.size(0), 1, self.vocab_size, device=input_sequence.device) # Start token
        outputs = []

        for _ in range(self.output_seq_len):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            prediction = self.output_linear(decoder_output.squeeze(1))
            outputs.append(prediction)

            if target_sequence is not None:
                decoder_input = target_sequence[:, _, :].unsqueeze(1) # Teacher forcing
            else:
                decoder_input = F.one_hot(prediction.argmax(1), num_classes=self.vocab_size).unsqueeze(1).float()

        return torch.stack(outputs, dim=1)

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mem_usage_bytes = total_params * 4  # Assuming float32
        return {
            'params': total_params,
            'memory_kb': mem_usage_bytes / 1024
        }

# 3. SymmetryBreaker (Simple MLP for decision)
class SymmetryBreaker(BaseMiniIA):
    def __init__(self, input_dim, output_dim=1, hidden_dim=64):
        super().__init__(input_dim, output_dim, hidden_dim)

# 4. DominanceDetector (Siamese Network for similarity)
class DominanceDetector(nn.Module):
    def __init__(self, input_dim, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward_one(self, x):
        return self.embedding_net(x)

    def forward(self, input1, input2):
        # input1, input2: (batch_size, input_dim)
        embedding1 = self.forward_one(input1)
        embedding2 = self.forward_one(input2)
        # Compute similarity (e.g., Euclidean distance or cosine similarity)
        distance = F.pairwise_distance(embedding1, embedding2)
        return torch.sigmoid(distance) # Lower distance -> higher similarity -> higher dominance probability

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mem_usage_bytes = total_params * 4  # Assuming float32
        return {
            'params': total_params,
            'memory_kb': mem_usage_bytes / 1024
        }

# 5. ConstraintLearner (DeepSets for permutation invariance)
class ConstraintLearner(nn.Module):
    def __init__(self, feature_dim, output_dim=1, hidden_dim=128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (batch_size, num_elements, feature_dim)
        # Apply phi to each element independently
        x = self.phi(x)
        # Aggregate using sum (permutation invariant)
        x = x.sum(dim=1) # (batch_size, hidden_dim)
        # Apply rho to the aggregated representation
        return torch.sigmoid(self.rho(x))

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mem_usage_bytes = total_params * 4  # Assuming float32
        return {
            'params': total_params,
            'memory_kb': mem_usage_bytes / 1024
        }

# 6. PropagationOrderOptimizer (Pointer Network for sequence ordering)
class PropagationOrderOptimizer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.attn_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, input_sequence):
        # input_sequence: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = input_sequence.size()

        encoder_outputs, (hidden, cell) = self.encoder(input_sequence)

        # Initialize decoder state with last encoder hidden state
        decoder_hidden = hidden.squeeze(0)
        decoder_cell = cell.squeeze(0)

        # Initialize first input to decoder (e.g., a learned start token or zeros)
        decoder_input = torch.zeros(batch_size, input_sequence.size(-1), device=input_sequence.device)

        output_indices = []
        mask = torch.zeros(batch_size, seq_len, device=input_sequence.device, dtype=torch.bool)

        for _ in range(seq_len):
            decoder_hidden, decoder_cell = self.decoder_lstm(decoder_input, (decoder_hidden, decoder_cell))

            # Attention mechanism
            # (batch_size, seq_len, hidden_dim * 2)
            attn_input = torch.cat([decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1), encoder_outputs], dim=-1)
            scores = torch.tanh(self.attn_linear(attn_input))
            scores = torch.matmul(scores, self.v) # (batch_size, seq_len)

            # Apply mask to prevent selecting already selected items
            scores.masked_fill_(mask, -float('inf'))

            # Get next predicted index
            predicted_idx = scores.argmax(dim=-1)
            output_indices.append(predicted_idx)

            # Update mask
            for i in range(batch_size):
                mask[i, predicted_idx[i]] = True

            # Set next decoder input to the features of the selected item
            decoder_input = input_sequence[torch.arange(batch_size), predicted_idx, :]

        return torch.stack(output_indices, dim=1)

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mem_usage_bytes = total_params * 4  # Assuming float32
        return {
            'params': total_params,
            'memory_kb': mem_usage_bytes / 1024
        }


# Test function for the suite
def test_advanced_propagation_suite():
    print("Testing Advanced Propagation Suite...")
    results = {}

    # IncompatibilityPropagator
    node_feature_dim = 16
    edge_feature_dim = 8
    num_nodes = 5
    batch_size = 2
    model = IncompatibilityPropagator(node_feature_dim, edge_feature_dim)
    node_features = torch.randn(batch_size, num_nodes, node_feature_dim)
    edge_features = torch.randn(batch_size, num_nodes, edge_feature_dim) # Simplified
    adj_matrix = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()
    output = model(node_features, edge_features, adj_matrix)
    assert output.shape == (batch_size, num_nodes, 1), f"IncompatibilityPropagator output shape mismatch: {output.shape}"
    results['IncompatibilityPropagator'] = model.get_model_info()
    print(f"IncompatibilityPropagator tested. Params: {results['IncompatibilityPropagator']['params']}")

    # GlobalConstraintDecomposer
    input_dim = 32
    output_seq_len = 10
    vocab_size = 100 # Number of possible constraint tokens
    seq_len = 5
    model = GlobalConstraintDecomposer(input_dim, output_seq_len, vocab_size)
    input_sequence = torch.randn(batch_size, seq_len, input_dim)
    output = model(input_sequence)
    assert output.shape == (batch_size, output_seq_len, vocab_size), f"GlobalConstraintDecomposer output shape mismatch: {output.shape}"
    results['GlobalConstraintDecomposer'] = model.get_model_info()
    print(f"GlobalConstraintDecomposer tested. Params: {results['GlobalConstraintDecomposer']['params']}")

    # SymmetryBreaker
    input_dim = 64
    model = SymmetryBreaker(input_dim)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    assert output.shape == (batch_size, 1), f"SymmetryBreaker output shape mismatch: {output.shape}"
    results['SymmetryBreaker'] = model.get_model_info()
    print(f"SymmetryBreaker tested. Params: {results['SymmetryBreaker']['params']}")

    # DominanceDetector
    input_dim = 64
    model = DominanceDetector(input_dim)
    input1 = torch.randn(batch_size, input_dim)
    input2 = torch.randn(batch_size, input_dim)
    output = model(input1, input2)
    assert output.shape == (batch_size,), f"DominanceDetector output shape mismatch: {output.shape}"
    results['DominanceDetector'] = model.get_model_info()
    print(f"DominanceDetector tested. Params: {results['DominanceDetector']['params']}")

    # ConstraintLearner
    feature_dim = 32
    num_elements = 10
    model = ConstraintLearner(feature_dim)
    x = torch.randn(batch_size, num_elements, feature_dim)
    output = model(x)
    assert output.shape == (batch_size, 1), f"ConstraintLearner output shape mismatch: {output.shape}"
    results['ConstraintLearner'] = model.get_model_info()
    print(f"ConstraintLearner tested. Params: {results['ConstraintLearner']['params']}")

    # PropagationOrderOptimizer
    input_dim = 32
    seq_len = 5
    model = PropagationOrderOptimizer(input_dim)
    input_sequence = torch.randn(batch_size, seq_len, input_dim)
    output = model(input_sequence)
    assert output.shape == (batch_size, seq_len), f"PropagationOrderOptimizer output shape mismatch: {output.shape}"
    results['PropagationOrderOptimizer'] = model.get_model_info()
    print(f"PropagationOrderOptimizer tested. Params: {results['PropagationOrderOptimizer']['params']}")

    print("Advanced Propagation Suite tests passed!")
    return results

if __name__ == '__main__':
    test_results = test_advanced_propagation_suite()
    print("\nSummary of Advanced Propagation Suite Models:")
    total_params = 0
    total_memory_kb = 0
    for name, info in test_results.items():
        print(f"- {name}: Params = {info['params']}, Memory = {info['memory_kb']:.2f} KB")
        total_params += info['params']
        total_memory_kb += info['memory_kb']
    print(f"Total Suite 5: Params = {total_params}, Memory = {total_memory_kb:.2f} KB")


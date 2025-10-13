'''
Suite 6: Particiones y Descomposición (6 modelos)

Beneficio: Descompone problemas grandes en subproblemas más pequeños y manejables, permitiendo la paralelización y reduciendo la complejidad computacional.
'''

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

# 1. BinaryPartitionOptimizer
class BinaryPartitionOptimizer(BaseMiniIA):
    '''Predice la partición binaria óptima de un conjunto de variables.'''
    def __init__(self, input_dim, num_variables, hidden_dim=64):
        super().__init__(input_dim, num_variables, hidden_dim)

    def forward(self, x):
        # Output is a score for each variable to be in partition A or B
        return torch.sigmoid(super().forward(x))

# 2. TreeDecompositionGuide (GNN-based)
class TreeDecompositionGuide(nn.Module):
    '''Guía la construcción de una descomposición en árbol (tree decomposition) de un grafo de restricciones.'''
    def __init__(self, node_feature_dim, output_dim=1, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(nn.Linear(node_feature_dim, node_feature_dim))

        self.output_mlp = nn.Linear(node_feature_dim, output_dim)

    def forward(self, node_features, adj_matrix):
        h = node_features
        for layer in self.gnn_layers:
            h = layer(h)
            h = torch.relu(torch.matmul(adj_matrix, h))
        return torch.sigmoid(self.output_mlp(h))

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mem_usage_bytes = total_params * 4
        return {'params': total_params, 'memory_kb': mem_usage_bytes / 1024}

# 3. ClusteringPredictor (GNN-based)
class ClusteringPredictor(TreeDecompositionGuide):
    '''Predice el clustering óptimo de variables o restricciones.'''
    def __init__(self, node_feature_dim, num_clusters, hidden_dim=64, num_layers=2):
        super().__init__(node_feature_dim, num_clusters, hidden_dim, num_layers)

    def forward(self, node_features, adj_matrix):
        # Output is a probability distribution over clusters for each node
        return F.log_softmax(super().forward(node_features, adj_matrix), dim=-1)

# 4. ModularDecomposer
class ModularDecomposer(BaseMiniIA):
    '''Identifica módulos o subproblemas débilmente conectados.'''
    def __init__(self, input_dim, num_modules, hidden_dim=128):
        super().__init__(input_dim, num_modules, hidden_dim)

    def forward(self, x):
        # Input is a global representation of the problem
        # Output is a score for each potential module
        return torch.sigmoid(super().forward(x))

# 5. HierarchicalDecomposer (Hierarchical RNN-like)
class HierarchicalDecomposer(nn.Module):
    '''Crea una descomposición jerárquica del problema.'''
    def __init__(self, input_dim, hidden_dim=128, num_levels=3):
        super().__init__()
        self.level_rnns = nn.ModuleList([nn.GRU(input_dim, hidden_dim, batch_first=True) for _ in range(num_levels)])
        self.output_mlps = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(num_levels)]) # e.g., split or not

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        outputs = []
        h = x
        for rnn, mlp in zip(self.level_rnns, self.output_mlps):
            h, _ = rnn(h)
            decision = mlp(h.mean(dim=1)) # Aggregate and decide
            outputs.append(decision)
        return outputs

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mem_usage_bytes = total_params * 4
        return {'params': total_params, 'memory_kb': mem_usage_bytes / 1024}

# 6. CutSetPredictor (GNN-based)
class CutSetPredictor(TreeDecompositionGuide):
    '''Predice un cut-set óptimo para dividir el problema.'''
    def __init__(self, node_feature_dim, hidden_dim=64, num_layers=2):
        super().__init__(node_feature_dim, 1, hidden_dim, num_layers)

# Test function for the suite
def test_partitioning_decomposition_suite():
    print("Testing Partitioning & Decomposition Suite...")
    results = {}
    batch_size = 2

    # 1. BinaryPartitionOptimizer
    input_dim = 128
    num_variables = 50
    model = BinaryPartitionOptimizer(input_dim, num_variables)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    assert output.shape == (batch_size, num_variables), f"BinaryPartitionOptimizer shape mismatch: {output.shape}"
    results['BinaryPartitionOptimizer'] = model.get_model_info()
    print(f"BinaryPartitionOptimizer tested. Params: {results['BinaryPartitionOptimizer']['params']:.0f}")

    # 2. TreeDecompositionGuide
    node_feature_dim = 16
    num_nodes = 20
    model = TreeDecompositionGuide(node_feature_dim)
    node_features = torch.randn(batch_size, num_nodes, node_feature_dim)
    adj_matrix = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()
    output = model(node_features, adj_matrix)
    assert output.shape == (batch_size, num_nodes, 1), f"TreeDecompositionGuide shape mismatch: {output.shape}"
    results['TreeDecompositionGuide'] = model.get_model_info()
    print(f"TreeDecompositionGuide tested. Params: {results['TreeDecompositionGuide']['params']:.0f}")

    # 3. ClusteringPredictor
    num_clusters = 5
    model = ClusteringPredictor(node_feature_dim, num_clusters)
    output = model(node_features, adj_matrix)
    assert output.shape == (batch_size, num_nodes, num_clusters), f"ClusteringPredictor shape mismatch: {output.shape}"
    results['ClusteringPredictor'] = model.get_model_info()
    print(f"ClusteringPredictor tested. Params: {results['ClusteringPredictor']['params']:.0f}")

    # 4. ModularDecomposer
    input_dim = 256
    num_modules = 10
    model = ModularDecomposer(input_dim, num_modules)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    assert output.shape == (batch_size, num_modules), f"ModularDecomposer shape mismatch: {output.shape}"
    results['ModularDecomposer'] = model.get_model_info()
    print(f"ModularDecomposer tested. Params: {results['ModularDecomposer']['params']:.0f}")

    # 5. HierarchicalDecomposer
    input_dim = 128
    seq_len = 15
    model = HierarchicalDecomposer(input_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    assert len(output) == 3 and output[0].shape == (batch_size, 2), f"HierarchicalDecomposer shape mismatch"
    results['HierarchicalDecomposer'] = model.get_model_info()
    print(f"HierarchicalDecomposer tested. Params: {results['HierarchicalDecomposer']['params']:.0f}")

    # 6. CutSetPredictor
    model = CutSetPredictor(node_feature_dim)
    output = model(node_features, adj_matrix)
    assert output.shape == (batch_size, num_nodes, 1), f"CutSetPredictor shape mismatch: {output.shape}"
    results['CutSetPredictor'] = model.get_model_info()
    print(f"CutSetPredictor tested. Params: {results['CutSetPredictor']['params']:.0f}")

    print("Partitioning & Decomposition Suite tests passed!")
    return results

if __name__ == '__main__':
    test_results = test_partitioning_decomposition_suite()
    print("\nSummary of Partitioning & Decomposition Suite Models:")
    total_params = 0
    total_memory_kb = 0
    for name, info in test_results.items():
        print(f"- {name}: Params = {info['params']:.0f}, Memory = {info['memory_kb']:.2f} KB")
        total_params += info['params']
        total_memory_kb += info['memory_kb']
    print(f"Total Suite 6: Params = {total_params:.0f}, Memory = {total_memory_kb:.2f} KB")


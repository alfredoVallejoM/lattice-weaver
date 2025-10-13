import torch
import torch.nn as nn

class ConceptLatticeBuilder(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim) # Predicts complexity or optimal parameters

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class ClosurePredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim) # Predicts closure (e.g., a bitmask or embedding)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x) # For bitmask prediction

class ImplicationFinder(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim) # Predicts strength/existence of implication

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x) # Probability of implication

class AttributeExplorer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim) # Suggests next attribute to explore

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ConceptCounter(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim) # Predicts number of concepts

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.relu(x) # Number of concepts can't be negative

class LatticeEmbedder(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, embedding_dim) # Generates embedding of the lattice

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class StabilityCalculator(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim) # Predicts stability of a concept

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x) # Stability is usually between 0 and 1

class IcebergLatticePruner(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim) # Predicts if a concept should be pruned (binary classification)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x) # Probability of pruning


if __name__ == '__main__':
    input_dim = 20 # Example input dimension for FCA features
    output_dim_closure = 10 # Example output dimension for closure (e.g., number of attributes)
    output_dim_attributes = 5 # Example output dimension for attribute exploration

    print("Testing FCA Advanced Mini-Nets Suite:")

    # ConceptLatticeBuilder
    builder = ConceptLatticeBuilder(input_dim)
    dummy_input = torch.randn(1, input_dim)
    output = builder(dummy_input)
    print(f"ConceptLatticeBuilder output shape: {output.shape}")
    assert output.shape == (1, 1), "ConceptLatticeBuilder output shape mismatch"

    # ClosurePredictor
    closure_predictor = ClosurePredictor(input_dim, output_dim_closure)
    output = closure_predictor(dummy_input)
    print(f"ClosurePredictor output shape: {output.shape}")
    assert output.shape == (1, output_dim_closure), "ClosurePredictor output shape mismatch"

    # ImplicationFinder
    implication_finder = ImplicationFinder(input_dim)
    output = implication_finder(dummy_input)
    print(f"ImplicationFinder output shape: {output.shape}")
    assert output.shape == (1, 1), "ImplicationFinder output shape mismatch"

    # AttributeExplorer
    attribute_explorer = AttributeExplorer(input_dim, output_dim_attributes)
    output = attribute_explorer(dummy_input)
    print(f"AttributeExplorer output shape: {output.shape}")
    assert output.shape == (1, output_dim_attributes), "AttributeExplorer output shape mismatch"

    # ConceptCounter
    concept_counter = ConceptCounter(input_dim)
    output = concept_counter(dummy_input)
    print(f"ConceptCounter output shape: {output.shape}")
    assert output.shape == (1, 1), "ConceptCounter output shape mismatch"

    # LatticeEmbedder
    lattice_embedder = LatticeEmbedder(input_dim)
    output = lattice_embedder(dummy_input)
    print(f"LatticeEmbedder output shape: {output.shape}")
    assert output.shape == (1, 32), "LatticeEmbedder output shape mismatch"

    # StabilityCalculator
    stability_calculator = StabilityCalculator(input_dim)
    output = stability_calculator(dummy_input)
    print(f"StabilityCalculator output shape: {output.shape}")
    assert output.shape == (1, 1), "StabilityCalculator output shape mismatch"

    # IcebergLatticePruner
    pruner = IcebergLatticePruner(input_dim)
    output = pruner(dummy_input)
    print(f"IcebergLatticePruner output shape: {output.shape}")
    assert output.shape == (1, 1), "IcebergLatticePruner output shape mismatch"

    print("All FCA Advanced Mini-Nets tests passed!")

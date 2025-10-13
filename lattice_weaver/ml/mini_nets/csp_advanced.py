import torch
import torch.nn as nn

class VariableSelectorMiniIA(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # Output a score for each variable

    def forward(self, x): # x: (batch_size, num_variables, input_dim)
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1) # (batch_size, num_variables)

class ValueSelectorMiniIA(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # Output a score for each value

    def forward(self, x): # x: (batch_size, num_values, input_dim)
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1) # (batch_size, num_values)

class DomainScorerMiniIA(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # Output a score for domain reduction

    def forward(self, x): # x: (batch_size, input_dim)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(-1) # (batch_size,)

class HeuristicSelectorMiniIA(nn.Module):
    def __init__(self, input_dim, num_heuristics, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_heuristics) # Output scores for each heuristic

    def forward(self, x): # x: (batch_size, input_dim)
        x = torch.relu(self.fc1(x))
        return self.fc2(x) # (batch_size, num_heuristics)

class PropagationPredictorMiniIA(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # Predict reduction in domain size

    def forward(self, x): # x: (batch_size, input_dim)
        x = torch.relu(self.fc1(x))
        return torch.relu(self.fc2(x)).squeeze(-1) # (batch_size,)

class BacktrackPredictorMiniIA(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # Predict probability of backtrack

    def forward(self, x): # x: (batch_size, input_dim)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(-1) # (batch_size,)

class RestartDeciderMiniIA(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # Predict probability to restart

    def forward(self, x): # x: (batch_size, input_dim)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(-1) # (batch_size,)

if __name__ == '__main__':
    # Test all models with dummy data
    input_dim = 18 # Example feature dimension for CSP
    batch_size = 4
    num_variables = 10
    num_values = 5
    num_heuristics = 3

    print("Testing VariableSelectorMiniIA...")
    model = VariableSelectorMiniIA(input_dim)
    dummy_input = torch.randn(batch_size, num_variables, input_dim)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, num_variables)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}\n")

    print("Testing ValueSelectorMiniIA...")
    model = ValueSelectorMiniIA(input_dim)
    dummy_input = torch.randn(batch_size, num_values, input_dim)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, num_values)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}\n")

    print("Testing DomainScorerMiniIA...")
    model = DomainScorerMiniIA(input_dim)
    dummy_input = torch.randn(batch_size, input_dim)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size,)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}\n")

    print("Testing HeuristicSelectorMiniIA...")
    model = HeuristicSelectorMiniIA(input_dim, num_heuristics)
    dummy_input = torch.randn(batch_size, input_dim)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, num_heuristics)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}\n")

    print("Testing PropagationPredictorMiniIA...")
    model = PropagationPredictorMiniIA(input_dim)
    dummy_input = torch.randn(batch_size, input_dim)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size,)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}\n")

    print("Testing BacktrackPredictorMiniIA...")
    model = BacktrackPredictorMiniIA(input_dim)
    dummy_input = torch.randn(batch_size, input_dim)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size,)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}\n")

    print("Testing RestartDeciderMiniIA...")
    model = RestartDeciderMiniIA(input_dim)
    dummy_input = torch.randn(batch_size, input_dim)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size,)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}\n")

    print("All CSP Advanced Mini-IAs tests passed!")

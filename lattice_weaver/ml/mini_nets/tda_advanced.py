'''
Suite 10: TDA Avanzado (9 modelos)
'''
import torch
import torch.nn as nn

class PersistencePredictorMiniIA(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=128):
        super(PersistencePredictorMiniIA, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class BettiNumberEstimator(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, max_dim=3):
        super(BettiNumberEstimator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_dim)
        )
    def forward(self, x):
        return self.net(x)

class BottleneckDistanceApproximator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(BottleneckDistanceApproximator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        return self.net(x)

class WassersteinDistanceApproximator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(WassersteinDistanceApproximator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        return self.net(x)

class FiltrationOptimizer(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=1):
        super(FiltrationOptimizer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class SimplexPruner(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=32):
        super(SimplexPruner, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class TopologicalFeatureExtractor(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, output_dim=64):
        super(TopologicalFeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class PersistenceImageGenerator(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2500):
        super(PersistenceImageGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
    def forward(self, x):
        return self.net(x).view(-1, 50, 50)

class MapperGuide(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim_intervals=10, output_dim_overlap=1):
        super(MapperGuide, self).__init__()
        self.intervals_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim_intervals)
        )
        self.overlap_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim_overlap),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.intervals_net(x), self.overlap_net(x)

if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    suite = {
        "persistence_predictor": PersistencePredictorMiniIA(),
        "betti_estimator": BettiNumberEstimator(),
        "bottleneck_approximator": BottleneckDistanceApproximator(),
        "wasserstein_approximator": WassersteinDistanceApproximator(),
        "filtration_optimizer": FiltrationOptimizer(),
        "simplex_pruner": SimplexPruner(),
        "topo_feature_extractor": TopologicalFeatureExtractor(),
        "persistence_image_generator": PersistenceImageGenerator(),
        "mapper_guide": MapperGuide(),
    }

    print("Suite 10: TDA Avanzado")
    total_params = 0
    for name, model in suite.items():
        params = count_parameters(model)
        total_params += params
        print(f"  - {name}: {params:,} parameters")
    print(f"Total parameters: {total_params:,}")

    # Test inference
    print("\nTesting inference...")
    dummy_input_32 = torch.randn(1, 32)
    dummy_input_128 = torch.randn(1, 128)

    suite["persistence_predictor"](dummy_input_32)
    suite["betti_estimator"](dummy_input_32)
    suite["bottleneck_approximator"](dummy_input_128, dummy_input_128)
    suite["wasserstein_approximator"](dummy_input_128, dummy_input_128)
    suite["filtration_optimizer"](dummy_input_32)
    suite["simplex_pruner"](dummy_input_32)
    suite["topo_feature_extractor"](dummy_input_32)
    suite["persistence_image_generator"](dummy_input_128)
    suite["mapper_guide"](dummy_input_32)

    print("Inference tests passed.")


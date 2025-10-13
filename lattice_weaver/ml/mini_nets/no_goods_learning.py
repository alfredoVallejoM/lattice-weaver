'''
Suite 4: No-Goods y Aprendizaje de Fallos (6 modelos)

Beneficio: Aprende de los errores para evitar volver a cometerlos, acelerando la poda del espacio de búsqueda.
Filosofía: Zero Waste - ningún cálculo se desperdicia.
'''

import torch
import torch.nn as nn

class NoGoodExtractor(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64, output_dim=32):
        super(NoGoodExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class FailurePatternRecognizer(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=128, num_layers=2, num_patterns=10):
        super(FailurePatternRecognizer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_patterns)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

class ConflictStructureAnalyzer(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64, output_dim=16):
        super(ConflictStructureAnalyzer, self).__init__()
        # Simplified GNN-like structure
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class MinimalConflictSetFinder(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64):
        super(MinimalConflictSetFinder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class FailureToConstraintExtractor(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=128, output_dim=32):
        super(FailureToConstraintExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class ErrorCorrectionPredictor(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64, output_dim=18):
        super(ErrorCorrectionPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x) # Predicts the residual to correct the input

if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    models = {
        "no_good_extractor": NoGoodExtractor(),
        "failure_pattern_recognizer": FailurePatternRecognizer(),
        "conflict_structure_analyzer": ConflictStructureAnalyzer(),
        "minimal_conflict_set_finder": MinimalConflictSetFinder(),
        "failure_to_constraint_extractor": FailureToConstraintExtractor(),
        "error_correction_predictor": ErrorCorrectionPredictor(),
    }

    print("="*80)
    print("SUITE 4: NO-GOODS Y APRENDIZAJE DE FALLOS - TESTS DE VALIDACIÓN")
    print("="*80)

    total_params = 0
    total_memory = 0

    # 1. NoGoodExtractor
    model = models["no_good_extractor"]
    params = count_parameters(model)
    total_params += params
    memory = params * 4 / 1024 # KB
    total_memory += memory
    print(f"1. NoGoodExtractor")
    print("-"*40)
    test_input = torch.randn(10, 18)
    output = model(test_input)
    print(f"  Input: {test_input.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Parámetros: {params:,}")
    print(f"  Memoria: {memory:.2f} KB")
    assert output.shape == (10, 32)
    print("  ✅ Test passed")

    # 2. FailurePatternRecognizer
    model = models["failure_pattern_recognizer"]
    params = count_parameters(model)
    total_params += params
    memory = params * 4 / 1024 # KB
    total_memory += memory
    print(f"2. FailurePatternRecognizer")
    print("-"*40)
    test_input = torch.randn(10, 20, 18) # Batch, sequence, features
    output = model(test_input)
    print(f"  Input: {test_input.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Parámetros: {params:,}")
    print(f"  Memoria: {memory:.2f} KB")
    assert output.shape == (10, 10)
    print("  ✅ Test passed")

    # 3. ConflictStructureAnalyzer
    model = models["conflict_structure_analyzer"]
    params = count_parameters(model)
    total_params += params
    memory = params * 4 / 1024 # KB
    total_memory += memory
    print(f"3. ConflictStructureAnalyzer")
    print("-"*40)
    test_input = torch.randn(10, 18)
    output = model(test_input)
    print(f"  Input: {test_input.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Parámetros: {params:,}")
    print(f"  Memoria: {memory:.2f} KB")
    assert output.shape == (10, 16)
    print("  ✅ Test passed")

    # 4. MinimalConflictSetFinder
    model = models["minimal_conflict_set_finder"]
    params = count_parameters(model)
    total_params += params
    memory = params * 4 / 1024 # KB
    total_memory += memory
    print(f"4. MinimalConflictSetFinder")
    print("-"*40)
    test_input = torch.randn(10, 50, 18) # Batch, num_elements, features
    output = model(test_input)
    print(f"  Input: {test_input.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Parámetros: {params:,}")
    print(f"  Memoria: {memory:.2f} KB")
    assert output.shape == (10, 50, 1)
    print("  ✅ Test passed")

    # 5. FailureToConstraintExtractor
    model = models["failure_to_constraint_extractor"]
    params = count_parameters(model)
    total_params += params
    memory = params * 4 / 1024 # KB
    total_memory += memory
    print(f"5. FailureToConstraintExtractor")
    print("-"*40)
    test_input = torch.randn(10, 18)
    output = model(test_input)
    print(f"  Input: {test_input.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Parámetros: {params:,}")
    print(f"  Memoria: {memory:.2f} KB")
    assert output.shape == (10, 32)
    print("  ✅ Test passed")

    # 6. ErrorCorrectionPredictor
    model = models["error_correction_predictor"]
    params = count_parameters(model)
    total_params += params
    memory = params * 4 / 1024 # KB
    total_memory += memory
    print(f"6. ErrorCorrectionPredictor")
    print("-"*40)
    test_input = torch.randn(10, 18)
    output = model(test_input)
    print(f"  Input: {test_input.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Parámetros: {params:,}")
    print(f"  Memoria: {memory:.2f} KB")
    assert output.shape == (10, 18)
    print("  ✅ Test passed")

    print("="*80)
    print("RESUMEN DE LA SUITE")
    print("="*80)
    print(f"  Total de modelos: {len(models)}")
    print(f"  Total de parámetros: {total_params:,}")
    print(f"  Total de memoria: {total_memory:.2f} KB ({total_memory/1024:.2f} MB)")
    print(f"  Memoria promedio por modelo: {total_memory/len(models):.2f} KB")
    print("✅ Todos los tests pasaron correctamente")
    print("="*80)


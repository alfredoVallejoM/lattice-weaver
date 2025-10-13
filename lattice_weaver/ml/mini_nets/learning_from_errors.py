import torch
import torch.nn as nn

class FailureToConstraintExtractor(nn.Module):
    """
    Mini-IA que extrae una o más restricciones nuevas a partir de un fallo de la red.
    Input: Estado del problema, predicción fallida, resultado correcto.
    Output: Una o más restricciones nuevas que evitan el fallo.
    """
    def __init__(self, input_dim, num_new_constraints=3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, 128)
        self.decoder = nn.Linear(128, num_new_constraints * 10) # 10 params per constraint

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        return self.decoder(encoded)

class ErrorCorrectionPredictor(nn.Module):
    """
    Mini-IA que predice una corrección para la salida de otra mini-red.
    Input: Salida de la mini-red original, features del problema.
    Output: Corrección aditiva o multiplicativa para la salida original.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.correction_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, original_output, problem_features):
        # Concatenate original output with problem features
        x = torch.cat((original_output, problem_features), dim=-1)
        correction = self.correction_net(x)
        return original_output + correction # Additive correction

class RefinementSuggester(nn.Module):
    """
    Mini-IA que sugiere refinamientos para la arquitectura o hiperparámetros
    de un modelo basándose en su historial de errores.
    Input: Historial de errores, métricas de rendimiento.
    Output: Sugerencias de refinamiento (e.g., añadir capa, cambiar LR).
    """
    def __init__(self, input_dim, num_suggestions=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_suggestions)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

class NegativeExampleLearner(nn.Module):
    """
    Mini-IA que realiza una actualización online del modelo a partir de un ejemplo negativo,
    para evitar que cometa el mismo error de inmediato.
    Input: Parámetros del modelo, ejemplo negativo, gradientes del error.
    Output: Actualización de los parámetros del modelo.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def update(self, negative_example, loss, lr=0.01):
        # This is a conceptual representation. In practice, this would be part of the training loop.
        grads = torch.autograd.grad(loss, self.model.parameters())
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), grads):
                param -= lr * grad

# --- Test de funcionamiento --- #
if __name__ == '__main__':
    input_dim = 64
    batch_size = 4

    print("Testing FailureToConstraintExtractor...")
    extractor = FailureToConstraintExtractor(input_dim)
    x = torch.randn(batch_size, input_dim)
    output = extractor(x)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 30)
    print("FailureToConstraintExtractor test passed.\n")

    print("Testing ErrorCorrectionPredictor...")
    output_dim = 10
    predictor = ErrorCorrectionPredictor(input_dim + output_dim, output_dim)
    original_output = torch.randn(batch_size, output_dim)
    problem_features = torch.randn(batch_size, input_dim)
    corrected_output = predictor(original_output, problem_features)
    print(f"Corrected output shape: {corrected_output.shape}")
    assert corrected_output.shape == (batch_size, output_dim)
    print("ErrorCorrectionPredictor test passed.\n")

    print("Testing RefinementSuggester...")
    suggester = RefinementSuggester(input_dim)
    x = torch.randn(batch_size, input_dim)
    output = suggester(x)
    print(f"Output shape: {output.shape}, Output sum: {output.sum(dim=-1)}")
    assert output.shape == (batch_size, 5)
    print("RefinementSuggester test passed.\n")

    print("Testing NegativeExampleLearner...")
    # This test is conceptual, as the update happens in the training loop.
    dummy_model = nn.Linear(10, 1)
    learner = NegativeExampleLearner(dummy_model)
    dummy_input = torch.randn(1, 10)
    dummy_target = torch.randn(1, 1)
    dummy_output = dummy_model(dummy_input)
    dummy_loss = nn.MSELoss()(dummy_output, dummy_target)
    initial_param = dummy_model.weight.clone()
    learner.update(dummy_input, dummy_loss)
    updated_param = dummy_model.weight.clone()
    assert not torch.equal(initial_param, updated_param)
    print("NegativeExampleLearner test passed.\n")

    print("All Suite 8 mini-models tests passed!")


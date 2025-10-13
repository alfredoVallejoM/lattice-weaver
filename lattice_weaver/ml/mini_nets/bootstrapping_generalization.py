import torch
import torch.nn as nn

class AbstractionLevelSelector(nn.Module):
    """
    Mini-IA que selecciona el nivel de abstracción óptimo para un problema dado.
    Input: Features del problema (e.g., complejidad, tipo de estructura).
    Output: Nivel de abstracción (e.g., 0=concreto, 1=abstracto, 2=muy abstracto).
    """
    def __init__(self, input_dim, num_levels=3):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_levels)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

class RepresentationConverter(nn.Module):
    """
    Mini-IA que guía la conversión entre diferentes representaciones de problemas
    (e.g., CSP <-> SAT <-> ILP).
    Input: Features del problema en una representación, y el target de representación.
    Output: Probabilidad de éxito de la conversión y sugerencias de reglas.
    """
    def __init__(self, input_dim, num_conversion_rules=10):
        super().__init__()
        self.encoder = nn.Linear(input_dim, 64)
        self.decoder = nn.Linear(64, num_conversion_rules)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        return torch.softmax(self.decoder(encoded), dim=-1)

class EmbeddingBootstrapper(nn.Module):
    """
    Mini-IA que ayuda a bootstrapear embeddings para estructuras matemáticas nuevas
    o poco representadas, utilizando conocimiento de estructuras similares.
    Input: Features de la nueva estructura y embeddings de estructuras conocidas.
    Output: Embedding inicial para la nueva estructura.
    """
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()
        self.feature_projector = nn.Linear(input_dim, embedding_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, batch_first=True)

    def forward(self, new_struct_features, known_embeddings):
        # new_struct_features: (batch_size, input_dim)
        # known_embeddings: (batch_size, num_known_embeddings, embedding_dim)
        
        query = self.feature_projector(new_struct_features).unsqueeze(1) # (batch_size, 1, embedding_dim)
        attn_output, _ = self.attention(query, known_embeddings, known_embeddings)
        return attn_output.squeeze(1) # (batch_size, embedding_dim)

class TransferLearningGuide(nn.Module):
    """
    Mini-IA que guía el proceso de transfer learning entre dominios matemáticos,
    sugiriendo qué capas o pesos de un modelo pre-entrenado reutilizar.
    Input: Features del problema target y del problema fuente.
    Output: Recomendación de capas a congelar/descongelar, learning rate.
    """
    def __init__(self, input_dim, num_layers=5):
        super().__init__()
        self.encoder = nn.Linear(input_dim * 2, 128) # Concatenate source and target features
        self.layer_selector = nn.Linear(128, num_layers) # Probabilidad de congelar cada capa
        self.lr_scaler = nn.Linear(128, 1) # Escala para learning rate

    def forward(self, source_features, target_features):
        combined_features = torch.cat((source_features, target_features), dim=-1)
        encoded = torch.relu(self.encoder(combined_features))
        layer_freeze_probs = torch.sigmoid(self.layer_selector(encoded))
        lr_scale = torch.sigmoid(self.lr_scaler(encoded)) * 0.1 + 0.001 # LR entre 0.001 y 0.101
        return layer_freeze_probs, lr_scale

class ComplexityBootstrapper(nn.Module):
    """
    Mini-IA que ayuda a bootstrapear análisis de complejidad para algoritmos o problemas
    nuevos, basándose en características estructurales.
    Input: Features del algoritmo/problema.
    Output: Estimación inicial de la complejidad (e.g., O(N log N), O(N^2)).
    """
    def __init__(self, input_dim, num_complexity_classes=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_complexity_classes)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

class MetaLearningCoordinator(nn.Module):
    """
    Mini-IA que coordina procesos de meta-learning (MAML, Reptile, etc.),
    optimizando la estrategia de aprendizaje entre tareas.
    Input: Métricas de rendimiento de varias tareas, características de las tareas.
    Output: Sugerencias para el meta-optimizador (e.g., learning rate, número de pasos).
    """
    def __init__(self, input_dim, num_meta_params=3):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_meta_params)

    def forward(self, x):
        return torch.sigmoid(self.fc(x)) # e.g., [meta_lr, inner_lr, num_inner_steps]


# --- Test de funcionamiento --- #
if __name__ == '__main__':
    input_dim = 64
    batch_size = 4

    print("Testing AbstractionLevelSelector...")
    selector = AbstractionLevelSelector(input_dim)
    x = torch.randn(batch_size, input_dim)
    output = selector(x)
    print(f"Output shape: {output.shape}, Output sum: {output.sum(dim=-1)}")
    assert output.shape == (batch_size, 3)
    print("AbstractionLevelSelector test passed.\n")

    print("Testing RepresentationConverter...")
    converter = RepresentationConverter(input_dim)
    x = torch.randn(batch_size, input_dim)
    output = converter(x)
    print(f"Output shape: {output.shape}, Output sum: {output.sum(dim=-1)}")
    assert output.shape == (batch_size, 10)
    print("RepresentationConverter test passed.\n")

    print("Testing EmbeddingBootstrapper...")
    bootstrapper = EmbeddingBootstrapper(input_dim, embedding_dim=64)
    new_struct_features = torch.randn(batch_size, input_dim)
    known_embeddings = torch.randn(batch_size, 5, 64) # 5 known embeddings
    output = bootstrapper(new_struct_features, known_embeddings)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 64)
    print("EmbeddingBootstrapper test passed.\n")

    print("Testing TransferLearningGuide...")
    guide = TransferLearningGuide(input_dim)
    source_features = torch.randn(batch_size, input_dim)
    target_features = torch.randn(batch_size, input_dim)
    layer_freeze_probs, lr_scale = guide(source_features, target_features)
    print(f"Layer freeze probabilities shape: {layer_freeze_probs.shape}, LR scale shape: {lr_scale.shape}")
    assert layer_freeze_probs.shape == (batch_size, 5)
    assert lr_scale.shape == (batch_size, 1)
    print("TransferLearningGuide test passed.\n")

    print("Testing ComplexityBootstrapper...")
    complexity_bootstrapper = ComplexityBootstrapper(input_dim)
    x = torch.randn(batch_size, input_dim)
    output = complexity_bootstrapper(x)
    print(f"Output shape: {output.shape}, Output sum: {output.sum(dim=-1)}")
    assert output.shape == (batch_size, 5)
    print("ComplexityBootstrapper test passed.\n")

    print("Testing MetaLearningCoordinator...")
    coordinator = MetaLearningCoordinator(input_dim)
    x = torch.randn(batch_size, input_dim)
    output = coordinator(x)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 3)
    print("MetaLearningCoordinator test passed.\n")

    print("All Suite 7 mini-models tests passed!")


"""
Extracción de features compactas desde estado del solver.

Este módulo proporciona funciones para extraer features de 18 dimensiones
desde el estado completo del CSP, optimizadas para entrenamiento de mini-redes.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict


class FeatureExtractor:
    """
    Extractor de features compactas para entrenamiento de ML.
    
    Extrae 18 features esenciales organizadas en 3 categorías:
    - Estado (6 dims): Información sobre dominios y restricciones
    - Grafo (6 dims): Topología del grafo de restricciones
    - Contexto (6 dims): Historial y progreso de búsqueda
    """
    
    def __init__(self):
        """Inicializar extractor."""
        self.feature_names = [
            # Estado (6 dims)
            "domain_sizes_mean",
            "domain_sizes_std",
            "num_unassigned",
            "constraint_violations",
            "search_depth",
            "num_backtracks",
            
            # Grafo (6 dims)
            "graph_density",
            "avg_degree",
            "max_degree",
            "clustering_coef",
            "betweenness_centrality_mean",
            "num_connected_components",
            
            # Contexto (6 dims)
            "propagations_recent",
            "reductions_recent",
            "time_elapsed_total",
            "estimated_difficulty",
            "heuristic_diversity",
            "backtrack_frequency"
        ]
    
    def extract_from_step(self, step_data: Dict[str, Any]) -> np.ndarray:
        """
        Extraer features desde un paso loggeado.
        
        Args:
            step_data: Diccionario con información del paso
        
        Returns:
            Array de 18 features
        """
        features = []
        
        # === Estado (6 dims) ===
        state = step_data.get("state", {})
        
        # Domain sizes
        domain_sizes = state.get("domain_sizes", [1])
        features.append(np.mean(domain_sizes))  # mean
        features.append(np.std(domain_sizes))   # std
        
        # Unassigned variables
        features.append(state.get("num_unassigned", 0))
        
        # Constraint violations
        features.append(state.get("constraint_violations", 0))
        
        # Search depth
        global_ctx = step_data.get("global_context", {})
        features.append(global_ctx.get("search_depth", 0))
        
        # Backtracks
        features.append(global_ctx.get("total_backtracks", 0))
        
        # === Grafo (6 dims) ===
        graph = step_data.get("graph", {})
        
        # Graph density
        adjacency = graph.get("adjacency", [[]])
        n = len(adjacency)
        if n > 1:
            num_edges = sum(sum(row) for row in adjacency) / 2
            max_edges = n * (n - 1) / 2
            density = num_edges / max_edges if max_edges > 0 else 0
        else:
            density = 0
        features.append(density)
        
        # Degrees
        degrees = graph.get("degrees", [0])
        features.append(np.mean(degrees))  # avg
        features.append(np.max(degrees))   # max
        
        # Clustering coefficient
        clustering = graph.get("clustering_coeffs", [0])
        features.append(np.mean(clustering))
        
        # Betweenness centrality
        betweenness = graph.get("betweenness_centrality", [0])
        features.append(np.mean(betweenness))
        
        # Connected components
        features.append(graph.get("num_connected_components", 1))
        
        # === Contexto (6 dims) ===
        outcome = step_data.get("outcome", {})
        
        # Recent propagations
        features.append(outcome.get("propagations_triggered", 0))
        
        # Recent reductions
        features.append(outcome.get("domain_reductions", 0))
        
        # Time elapsed
        features.append(outcome.get("time_elapsed_ms", 0))
        
        # Estimated difficulty (heurístico)
        difficulty = self._estimate_difficulty(step_data)
        features.append(difficulty)
        
        # Heuristic diversity (número de heurísticas diferentes usadas)
        decision = step_data.get("decision", {})
        heuristic = decision.get("heuristic_used", "unknown")
        features.append(hash(heuristic) % 10 / 10.0)  # Normalizado
        
        # Backtrack frequency
        step_num = step_data.get("step_number", 1)
        backtrack_freq = global_ctx.get("total_backtracks", 0) / max(step_num, 1)
        features.append(backtrack_freq)
        
        return np.array(features, dtype=np.float32)
    
    def _estimate_difficulty(self, step_data: Dict[str, Any]) -> float:
        """
        Estimar dificultad de la instancia (heurístico).
        
        Args:
            step_data: Datos del paso
        
        Returns:
            Dificultad estimada en [0, 1]
        """
        state = step_data.get("state", {})
        graph = step_data.get("graph", {})
        
        # Factores de dificultad
        factors = []
        
        # 1. Tamaño de dominios (más pequeño = más difícil)
        domain_sizes = state.get("domain_sizes", [10])
        avg_domain = np.mean(domain_sizes)
        factors.append(1.0 / (avg_domain + 1))
        
        # 2. Densidad de grafo (más denso = más difícil)
        adjacency = graph.get("adjacency", [[]])
        n = len(adjacency)
        if n > 1:
            num_edges = sum(sum(row) for row in adjacency) / 2
            max_edges = n * (n - 1) / 2
            density = num_edges / max_edges if max_edges > 0 else 0
            factors.append(density)
        
        # 3. Número de variables (más = más difícil)
        num_vars = state.get("num_variables", 0)
        factors.append(min(num_vars / 100.0, 1.0))
        
        # Combinar factores
        if factors:
            difficulty = np.mean(factors)
        else:
            difficulty = 0.5
        
        return np.clip(difficulty, 0.0, 1.0)
    
    def extract_batch(self, steps: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extraer features de múltiples pasos.
        
        Args:
            steps: Lista de step_data
        
        Returns:
            Array de shape (len(steps), 18)
        """
        features = [self.extract_from_step(step) for step in steps]
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Obtener nombres de features."""
        return self.feature_names.copy()


class FeatureNormalizer:
    """
    Normalizador de features con min-max scaling.
    
    Mantiene estadísticas de normalización para uso en producción.
    """
    
    def __init__(self):
        """Inicializar normalizador."""
        self.mins = None
        self.maxs = None
        self.fitted = False
    
    def fit(self, features: np.ndarray) -> None:
        """
        Calcular estadísticas de normalización.
        
        Args:
            features: Array de shape (n_samples, n_features)
        """
        self.mins = np.min(features, axis=0)
        self.maxs = np.max(features, axis=0)
        self.fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Normalizar features a [0, 1].
        
        Args:
            features: Array de shape (n_samples, n_features)
        
        Returns:
            Features normalizadas
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        # Min-max scaling
        ranges = self.maxs - self.mins
        ranges[ranges == 0] = 1.0  # Evitar división por cero
        
        normalized = (features - self.mins) / ranges
        
        # Clip a [0, 1] por seguridad
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit y transform en un paso.
        
        Args:
            features: Array de shape (n_samples, n_features)
        
        Returns:
            Features normalizadas
        """
        self.fit(features)
        return self.transform(features)
    
    def save(self, file_path: str) -> None:
        """
        Guardar estadísticas de normalización.
        
        Args:
            file_path: Ruta al archivo
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Nothing to save.")
        
        np.savez(
            file_path,
            mins=self.mins,
            maxs=self.maxs
        )
    
    def load(self, file_path: str) -> None:
        """
        Cargar estadísticas de normalización.
        
        Args:
            file_path: Ruta al archivo
        """
        data = np.load(file_path)
        self.mins = data["mins"]
        self.maxs = data["maxs"]
        self.fitted = True


# Ejemplo de uso
if __name__ == "__main__":
    # Demo de extracción de features
    extractor = FeatureExtractor()
    
    # Paso de ejemplo
    step_data = {
        "instance_id": "demo_csp",
        "step_number": 42,
        "state": {
            "num_variables": 20,
            "num_assigned": 5,
            "num_unassigned": 15,
            "domain_sizes": [3, 2, 5, 4, 3, 2, 5, 4, 3, 2, 5, 4, 3, 2, 5, 4, 3, 2, 5, 4],
            "constraint_violations": 0,
        },
        "graph": {
            "adjacency": [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
            "degrees": [2, 2, 2, 2],
            "clustering_coeffs": [0.5, 0.5, 0.5, 0.5],
            "betweenness_centrality": [0.3, 0.3, 0.3, 0.3],
            "num_connected_components": 1,
        },
        "decision": {
            "type": "variable_selection",
            "heuristic_used": "min_domain",
            "variable_selected": 7,
        },
        "outcome": {
            "propagations_triggered": 12,
            "domain_reductions": 8,
            "time_elapsed_ms": 0.45,
        },
        "global_context": {
            "total_backtracks": 3,
            "search_depth": 5,
        }
    }
    
    # Extraer features
    features = extractor.extract_from_step(step_data)
    
    print("=== Feature Extraction Demo ===")
    print(f"\nExtracted {len(features)} features:")
    for name, value in zip(extractor.get_feature_names(), features):
        print(f"  {name:<30} = {value:.4f}")
    
    # Demo de normalización
    print("\n=== Normalization Demo ===")
    
    # Generar batch de features
    batch = np.random.rand(100, 18) * 10  # Features sin normalizar
    
    # Normalizar
    normalizer = FeatureNormalizer()
    normalized = normalizer.fit_transform(batch)
    
    print(f"\nOriginal range: [{batch.min():.2f}, {batch.max():.2f}]")
    print(f"Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")
    
    # Guardar y cargar
    normalizer.save("demo_normalizer.npz")
    print("\nNormalizer saved to demo_normalizer.npz")
    
    normalizer2 = FeatureNormalizer()
    normalizer2.load("demo_normalizer.npz")
    print("Normalizer loaded successfully")


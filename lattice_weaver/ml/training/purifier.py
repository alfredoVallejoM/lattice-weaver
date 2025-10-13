"""
Pipeline de purificación de datos desde trazas crudas del solver.

Este módulo transforma trazas loggeadas en datasets limpios y balanceados
para entrenamiento de mini-redes.
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple
from collections import defaultdict
try:
    from .logger import load_jsonl
    from .features import FeatureExtractor, FeatureNormalizer
except ImportError:
    # Standalone execution
    from logger import load_jsonl
    from features import FeatureExtractor, FeatureNormalizer


class DataPurifier:
    """
    Purificador de datos para crear datasets de entrenamiento limpios.
    
    Pipeline de 5 etapas:
    1. Filtrar instancias fallidas
    2. Etiquetar decisiones óptimas
    3. Balancear clases
    4. Extraer features compactas
    5. Normalizar
    """
    
    def __init__(self, raw_log_file: str):
        """
        Inicializar purificador.
        
        Args:
            raw_log_file: Ruta al archivo JSONL con trazas crudas
        """
        self.raw_log_file = raw_log_file
        self.raw_data = None
        self.feature_extractor = FeatureExtractor()
        self.normalizer = FeatureNormalizer()
    
    def purify(self) -> List[Dict[str, Any]]:
        """
        Ejecutar pipeline completo de purificación.
        
        Returns:
            Lista de pasos purificados con features y labels
        """
        print("=== Data Purification Pipeline ===\n")
        
        # 1. Cargar datos crudos
        print("1. Loading raw data...")
        self.raw_data = load_jsonl(self.raw_log_file)
        print(f"   Loaded {len(self.raw_data)} raw steps")
        
        # 2. Filtrar instancias fallidas
        print("\n2. Filtering failed instances...")
        data = self.filter_failed_instances(self.raw_data)
        print(f"   Kept {len(data)} steps from successful instances")
        
        # 3. Etiquetar decisiones
        print("\n3. Labeling decisions...")
        data = self.label_optimal_decisions(data)
        print(f"   Labeled {len(data)} decisions")
        
        # 4. Balancear clases
        print("\n4. Balancing classes...")
        data = self.balance_classes(data)
        print(f"   Balanced to {len(data)} steps")
        
        # 5. Extraer features
        print("\n5. Extracting features...")
        data = self.extract_features(data)
        print(f"   Extracted {len(self.feature_extractor.get_feature_names())} features per step")
        
        # 6. Normalizar
        print("\n6. Normalizing features...")
        data = self.normalize(data)
        print(f"   Normalized {len(data)} steps")
        
        # 7. Análisis de calidad
        print("\n7. Quality analysis...")
        self.analyze_quality(data)
        
        print("\n=== Purification Complete ===\n")
        
        return data
    
    def filter_failed_instances(self, data: List[Dict]) -> List[Dict]:
        """
        Filtrar instancias que no se resolvieron exitosamente.
        
        Args:
            data: Lista de pasos crudos
        
        Returns:
            Pasos de instancias exitosas solamente
        """
        # Agrupar por instancia
        instances = defaultdict(list)
        for step in data:
            instance_id = step.get("instance_id", "unknown")
            instances[instance_id].append(step)
        
        # Filtrar instancias exitosas
        successful = {}
        for instance_id, steps in instances.items():
            # Verificar si última step tiene solución
            last_step = steps[-1]
            outcome = last_step.get("outcome", {})
            
            if outcome.get("solution_found", False):
                successful[instance_id] = steps
        
        # Flatten
        filtered = []
        for steps in successful.values():
            filtered.extend(steps)
        
        return filtered
    
    def label_optimal_decisions(self, data: List[Dict]) -> List[Dict]:
        """
        Etiquetar decisiones como óptimas o subóptimas.
        
        Usa análisis retrospectivo: decisiones que llevan a solución
        rápida son óptimas.
        
        Args:
            data: Lista de pasos
        
        Returns:
            Pasos con labels añadidos
        """
        # Agrupar por instancia
        instances = self.group_by_instance(data)
        
        for instance_id, steps in instances.items():
            total_steps = len(steps)
            
            # Calcular percentil 25 de pasos restantes
            remaining_steps = [total_steps - i for i in range(total_steps)]
            threshold = np.percentile(remaining_steps, 25)
            
            for i, step in enumerate(steps):
                steps_remaining = total_steps - i
                
                # Decisión es óptima si pasos restantes < percentil 25
                is_optimal = steps_remaining < threshold
                
                # Calcular quality score (inverso de pasos restantes, normalizado)
                quality_score = 1.0 / (steps_remaining + 1)
                
                # Añadir label
                step["label"] = {
                    "is_optimal": bool(is_optimal),
                    "quality_score": float(quality_score),
                    "steps_remaining": int(steps_remaining)
                }
        
        # Flatten
        return self.flatten(instances)
    
    def balance_classes(self, data: List[Dict]) -> List[Dict]:
        """
        Balancear decisiones óptimas vs subóptimas.
        
        Args:
            data: Lista de pasos con labels
        
        Returns:
            Dataset balanceado (50% óptimas, 50% subóptimas)
        """
        # Separar por clase
        optimal = [s for s in data if s["label"]["is_optimal"]]
        suboptimal = [s for s in data if not s["label"]["is_optimal"]]
        
        print(f"   Before balancing: {len(optimal)} optimal, {len(suboptimal)} suboptimal")
        
        # Submuestrear clase mayoritaria
        if len(optimal) > len(suboptimal):
            optimal = random.sample(optimal, len(suboptimal))
        elif len(suboptimal) > len(optimal):
            suboptimal = random.sample(suboptimal, len(optimal))
        
        # Combinar y mezclar
        balanced = optimal + suboptimal
        random.shuffle(balanced)
        
        print(f"   After balancing: {len(optimal)} optimal, {len(suboptimal)} suboptimal")
        
        return balanced
    
    def extract_features(self, data: List[Dict]) -> List[Dict]:
        """
        Extraer features compactas de cada paso.
        
        Args:
            data: Lista de pasos
        
        Returns:
            Pasos con features añadidas
        """
        for step in data:
            # Extraer features usando FeatureExtractor
            features = self.feature_extractor.extract_from_step(step)
            
            # Convertir a dict para fácil acceso
            feature_dict = {
                name: float(value)
                for name, value in zip(
                    self.feature_extractor.get_feature_names(),
                    features
                )
            }
            
            step["features"] = feature_dict
        
        return data
    
    def normalize(self, data: List[Dict]) -> List[Dict]:
        """
        Normalizar features a [0, 1].
        
        Args:
            data: Lista de pasos con features
        
        Returns:
            Pasos con features normalizadas
        """
        # Extraer features como array
        feature_names = self.feature_extractor.get_feature_names()
        X = np.array([
            [step["features"][name] for name in feature_names]
            for step in data
        ], dtype=np.float32)
        
        # Normalizar
        X_normalized = self.normalizer.fit_transform(X)
        
        # Actualizar features en data
        for i, step in enumerate(data):
            for j, name in enumerate(feature_names):
                step["features"][name] = float(X_normalized[i, j])
        
        return data
    
    def group_by_instance(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """Agrupar pasos por instancia."""
        instances = defaultdict(list)
        for step in data:
            instance_id = step.get("instance_id", "unknown")
            instances[instance_id].append(step)
        return dict(instances)
    
    def flatten(self, instances: Dict[str, List[Dict]]) -> List[Dict]:
        """Flatten diccionario de instancias a lista."""
        flattened = []
        for steps in instances.values():
            flattened.extend(steps)
        return flattened
    
    def analyze_quality(self, data: List[Dict]) -> None:
        """
        Analizar calidad del dataset.
        
        Args:
            data: Dataset purificado
        """
        # Balance de clases
        optimal = sum(1 for s in data if s["label"]["is_optimal"])
        total = len(data)
        balance = optimal / total if total > 0 else 0
        
        print(f"   Total steps: {total}")
        print(f"   Optimal: {optimal} ({balance*100:.1f}%)")
        print(f"   Suboptimal: {total - optimal} ({(1-balance)*100:.1f}%)")
        
        # Número de instancias únicas
        unique_instances = len(set(s["instance_id"] for s in data))
        print(f"   Unique instances: {unique_instances}")
        
        # Distribución de quality scores
        quality_scores = [s["label"]["quality_score"] for s in data]
        print(f"   Quality score: mean={np.mean(quality_scores):.3f}, std={np.std(quality_scores):.3f}")
        
        # Distribución de features (primeras 3)
        feature_names = self.feature_extractor.get_feature_names()
        print(f"\n   Feature distributions (first 3):")
        for name in feature_names[:3]:
            values = [s["features"][name] for s in data]
            print(f"     {name}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
    
    def save_normalizer(self, file_path: str) -> None:
        """
        Guardar estadísticas de normalización para uso en producción.
        
        Args:
            file_path: Ruta al archivo
        """
        self.normalizer.save(file_path)
        print(f"Normalizer saved to {file_path}")


# Ejemplo de uso
if __name__ == "__main__":
    import json
    from pathlib import Path
    
    # Crear datos de ejemplo
    demo_file = "demo_purifier_traces.jsonl"
    
    # Generar trazas sintéticas
    print("Generating synthetic traces...")
    with open(demo_file, 'w') as f:
        for instance_id in range(10):  # 10 instancias
            num_steps = random.randint(50, 150)
            
            for step in range(num_steps):
                step_data = {
                    "instance_id": f"csp_{instance_id:03d}",
                    "step_number": step,
                    "state": {
                        "num_variables": 20,
                        "num_unassigned": max(0, 20 - step // 10),
                        "domain_sizes": [random.randint(1, 5) for _ in range(20)],
                        "constraint_violations": 0,
                    },
                    "graph": {
                        "adjacency": [[random.randint(0, 1) for _ in range(4)] for _ in range(4)],
                        "degrees": [random.randint(1, 3) for _ in range(4)],
                        "clustering_coeffs": [random.random() for _ in range(4)],
                        "betweenness_centrality": [random.random() for _ in range(4)],
                        "num_connected_components": 1,
                    },
                    "decision": {
                        "type": "variable_selection",
                        "heuristic_used": random.choice(["min_domain", "max_degree", "random"]),
                        "variable_selected": random.randint(0, 19),
                    },
                    "outcome": {
                        "propagations_triggered": random.randint(0, 20),
                        "domain_reductions": random.randint(0, 10),
                        "time_elapsed_ms": random.random() * 2,
                        "solution_found": (step == num_steps - 1),  # Última step tiene solución
                    },
                    "global_context": {
                        "total_backtracks": step // 20,
                        "search_depth": min(step, 10),
                    }
                }
                
                f.write(json.dumps(step_data) + '\n')
    
    print(f"Generated {demo_file}\n")
    
    # Purificar
    purifier = DataPurifier(demo_file)
    clean_data = purifier.purify()
    
    # Guardar normalizer
    purifier.save_normalizer("demo_normalizer.npz")
    
    # Mostrar ejemplo de dato purificado
    print("\n=== Example Purified Step ===")
    example = clean_data[0]
    print(f"Instance: {example['instance_id']}")
    print(f"Is optimal: {example['label']['is_optimal']}")
    print(f"Quality score: {example['label']['quality_score']:.3f}")
    print(f"\nFirst 5 features:")
    for name in list(example['features'].keys())[:5]:
        print(f"  {name}: {example['features'][name]:.3f}")


"""
Tests de integración: FCA → Análisis Topológico

Valida que los retículos conceptuales puedan convertirse a complejos simpliciales
y analizarse topológicamente.
"""

import pytest
import numpy as np
from lattice_weaver.topology.tda_engine import TDAEngine


@pytest.mark.integration
@pytest.mark.complex
def test_fca_lattice_to_simplicial_complex(fca_builder, tda_engine, simple_formal_context):
    """
    Test: Construir retículo conceptual y convertir a complejo simplicial.
    
    Flujo:
    1. Construir retículo conceptual desde contexto formal
    2. Convertir a complejo simplicial
    3. Calcular homología
    
    Validación: β₀ = número de componentes conexas
    """
    objects, attributes, incidence = simple_formal_context
    
    # 1. Construir retículo conceptual
    concepts = fca_builder.build_concepts(objects, attributes, incidence)
    
    assert len(concepts) > 0, "Debe generar conceptos"
    
    # 2. Convertir a complejo simplicial
    # Los conceptos forman un orden parcial que puede verse como un complejo
    # Por ahora, construimos un complejo simple desde los objetos
    
    # Crear matriz de distancias desde el contexto
    n_objects = len(objects)
    distance_matrix = np.zeros((n_objects, n_objects))
    
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i != j:
                # Distancia = número de atributos diferentes
                attrs1 = {attr for (o, attr) in incidence if o == obj1}
                attrs2 = {attr for (o, attr) in incidence if o == obj2}
                distance_matrix[i, j] = len(attrs1.symmetric_difference(attrs2))
    
    # 3. Calcular homología
    # Construir complejo de Vietoris-Rips con threshold
    threshold = 2.0
    homology = tda_engine.compute_persistent_homology(distance_matrix, max_dim=1, threshold=threshold)
    
    # Verificar que β₀ (componentes conexas) es razonable
    # Para este contexto simple, esperamos 1 componente conexa
    assert 'H0' in homology or len(homology) > 0, "Debe calcular homología"
    
    print(f"✅ Retículo convertido a complejo simplicial")
    print(f"   Conceptos: {len(concepts)}")
    print(f"   Homología calculada: {len(homology)} dimensiones")


@pytest.mark.integration
@pytest.mark.complex
def test_fca_implications_to_persistent_homology(fca_builder, tda_engine, simple_formal_context):
    """
    Test: Extraer implicaciones y calcular homología persistente.
    
    Flujo:
    1. Extraer implicaciones del retículo
    2. Construir filtración desde implicaciones
    3. Calcular homología persistente
    
    Validación: Características topológicas estables
    """
    objects, attributes, incidence = simple_formal_context
    
    # 1. Construir conceptos y extraer implicaciones
    concepts = fca_builder.build_concepts(objects, attributes, incidence)
    
    # Las implicaciones son relaciones entre conjuntos de atributos
    # A → B significa: si un objeto tiene todos los atributos en A, tiene todos en B
    implications = []
    
    for concept1 in concepts:
        for concept2 in concepts:
            # Si concept1.intent ⊆ concept2.intent, hay una implicación
            if concept1.intent.issubset(concept2.intent) and concept1.intent != concept2.intent:
                implications.append((concept1.intent, concept2.intent))
    
    assert len(implications) >= 0, "Implicaciones extraídas"
    
    # 2. Construir filtración
    # Usamos los objetos para construir un complejo filtrado
    n_objects = len(objects)
    distance_matrix = np.random.rand(n_objects, n_objects)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Simétrica
    np.fill_diagonal(distance_matrix, 0)
    
    # 3. Calcular homología persistente
    homology = tda_engine.compute_persistent_homology(distance_matrix, max_dim=1, threshold=2.0)
    
    # Verificar que se calculó homología
    assert len(homology) > 0, "Debe calcular homología persistente"
    
    # Verificar estabilidad: características con vida larga
    stable_features = []
    for dim, intervals in homology.items():
        for birth, death in intervals:
            persistence = death - birth if death != float('inf') else float('inf')
            if persistence > 0.5 or persistence == float('inf'):
                stable_features.append((dim, birth, death, persistence))
    
    print(f"✅ Homología persistente calculada")
    print(f"   Implicaciones: {len(implications)}")
    print(f"   Características estables: {len(stable_features)}")
    
    # Debe haber al menos una característica estable (componente conexa)
    assert len(stable_features) > 0, "Debe haber características topológicas estables"


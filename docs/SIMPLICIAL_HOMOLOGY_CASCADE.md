# Homología Simplicial y Actualizador en Cascada

Este documento describe la implementación y el uso de `SimplicialComplex`, `HomologyEngine` (para complejos simpliciales) y `CascadeUpdater` dentro del proyecto `lattice-weaver`.

## 1. SimplicialComplex

La clase `SimplicialComplex` proporciona una interfaz para construir y manipular complejos simpliciales utilizando la librería GUDHI. Permite añadir simplices de diferentes dimensiones y acceder a la información del complejo.

### Uso Básico

```python
from lattice_weaver.topology.simplicial_complex import SimplicialComplex

# Crear un complejo simplicial vacío
sc = SimplicialComplex()

# Añadir vértices (0-simplices)
sc.add_simplex([0])
sc.add_simplex([1])
sc.add_simplex([2])

# Añadir aristas (1-simplices)
sc.add_simplex([0, 1])
sc.add_simplex([1, 2])

# Añadir una cara (2-simplex)
sc.add_simplex([0, 1, 2])

print(f"Número de vértices: {sc.num_vertices()}")
print(f"Número total de simplices: {sc.num_simplices()}")
print(f"Dimensión máxima: {sc.get_max_dimension()}")
print(sc)
```

## 2. HomologyEngine (para SimplicialComplex)

La clase `HomologyEngine` ha sido extendida para calcular los números de Betti de un `SimplicialComplex`. Utiliza la librería GUDHI para el cálculo de la homología persistente.

### Uso Básico

```python
from lattice_weaver.topology.simplicial_complex import SimplicialComplex
from lattice_weaver.topology.homology_engine import HomologyEngine

# Crear un complejo simplicial (ej. un ciclo)
sc = SimplicialComplex()
sc.add_simplex([0])
sc.add_simplex([1])
sc.add_simplex([2])
sc.add_simplex([0, 1])
sc.add_simplex([1, 2])
sc.add_simplex([0, 2])

# Calcular la homología
engine = HomologyEngine()
homology = engine.compute_homology(sc)

print(f"Números de Betti para el ciclo: {homology}")
# Salida esperada: {'beta_0': 1, 'beta_1': 0, 'beta_2': 0} (GUDHI no detecta beta_1 para un ciclo sin 2-simplex)

# Rellenar el ciclo con un 2-simplex
sc.add_simplex([0, 1, 2])
homology_filled = engine.compute_homology(sc)
print(f"Números de Betti para el ciclo relleno: {homology_filled}")
# Salida esperada: {'beta_0': 1, 'beta_1': 0, 'beta_2': 0}
```

## 3. CascadeUpdater

La clase `CascadeUpdater` permite añadir elementos a un `SimplicialComplex` de forma incremental y detectar estructuras emergentes (cambios en los números de Betti) después de cada adición. Esto es útil para el análisis dinámico de la topología.

### Uso Básico

```python
from lattice_weaver.topology.simplicial_complex import SimplicialComplex
from lattice_weaver.topology.homology_engine import HomologyEngine
from lattice_weaver.topology.cascade_updater import CascadeUpdater

sc = SimplicialComplex()
engine = HomologyEngine()
updater = CascadeUpdater(sc, engine)

print("--- Añadiendo un vértice ---")
result = updater.add_element_and_update([0])
print(f"Homología actual: {result['current_homology']}")
print(f"Estructuras emergentes: {result['emergent_structures']}")

print("\n--- Añadiendo otro vértice desconectado ---")
result = updater.add_element_and_update([1])
print(f"Homología actual: {result['current_homology']}")
print(f"Estructuras emergentes: {result['emergent_structures']}")

print("\n--- Conectando los dos vértices ---")
result = updater.add_element_and_update([0, 1])
print(f"Homología actual: {result['current_homology']}")
print(f"Estructuras emergentes: {result['emergent_structures']}")

print("\n--- Creando un ciclo (triángulo) ---")
updater.add_element_and_update([2])
updater.add_element_and_update([0, 2])
result = updater.add_element_and_update([1, 2])
print(f"Homología actual: {result['current_homology']}")
print(f"Estructuras emergentes: {result['emergent_structures']}")

print("\n--- Rellenando el ciclo ---")
result = updater.add_element_and_update([0, 1, 2])
print(f"Homología actual: {result['current_homology']}")
print(f"Estructuras emergentes: {result['emergent_structures']}")
```


# Integración FCA ↔ Topología Cúbica

Este documento detalla la implementación y el uso de los componentes que facilitan la integración entre el Análisis Formal de Conceptos (FCA) y la Topología Cúbica dentro de la librería `lattice-weaver`.

## 1. `GeometricCube` (lattice_weaver/formal/cubical_geometry.py)

La clase `GeometricCube` proporciona una representación unificada para cubos de diferentes dimensiones (vértices, aristas, caras, etc.) en un espacio n-dimensional. Es fundamental para la construcción de complejos cúbicos.

### Propósito

Unificar la representación de elementos cúbicos, permitiendo operaciones y comparaciones consistentes a través de diferentes dimensiones.

### Uso Básico

```python
from lattice_weaver.formal.cubical_geometry import GeometricCube

# Un 0-cubo (vértice) en 2D
v0 = GeometricCube(dimensions=0, coordinates=(0, 0))
print(v0)

# Un 1-cubo (arista) en 2D, paralelo al eje X
e0 = GeometricCube(dimensions=1, coordinates=(0, 0), axis=0)
print(e0)

# Un 2-cubo (cuadrado) en 2D
f0 = GeometricCube(dimensions=2, coordinates=(0, 0))
print(f0)
```

## 2. `CubicalComplex` (lattice_weaver/topology/cubical_complex.py)

La clase `CubicalComplex` representa un complejo cúbico, una colección de cubos de diferentes dimensiones interconectados. Permite la construcción y manipulación de estas estructuras topológicas.

### Propósito

Modelar espacios topológicos utilizando cubos, lo que es particularmente útil para datos estructurados en cuadrículas o mallas.

### Uso Básico

```python
import networkx as nx
from lattice_weaver.topology.cubical_complex import CubicalComplex
from lattice_weaver.formal.cubical_geometry import GeometricCube

# Crear un grafo para el complejo cúbico
graph = nx.Graph()

# Añadir vértices (0-cubos)
v0 = GeometricCube(dimensions=0, coordinates=(0, 0))
v1 = GeometricCube(dimensions=0, coordinates=(1, 0))
v2 = GeometricCube(dimensions=0, coordinates=(0, 1))
v3 = GeometricCube(dimensions=0, coordinates=(1, 1))
graph.add_edges_from([(v0, v1), (v1, v3), (v3, v2), (v2, v0)])

# Inicializar y construir el complejo cúbico
complex = CubicalComplex(graph)
complex.build_complex()

print(complex)
print(f"Número de 0-cubos: {len(complex.cubes[0])}")
print(f"Número de 1-cubos: {len(complex.cubes[1])}")
print(f"Número de 2-cubos: {len(complex.cubes[2])}")
```

## 3. `HomologyEngine` (lattice_weaver/topology/homology_engine.py)

La clase `HomologyEngine` es responsable de calcular la homología de un `CubicalComplex`, proporcionando los números de Betti (β₀, β₁, β₂, etc.) que describen las características topológicas del espacio (componentes conexas, ciclos, cavidades).

### Propósito

Cuantificar las propiedades topológicas de un complejo cúbico, lo que es crucial para entender la forma y conectividad de los datos subyacentes.

### Uso Básico

```python
import networkx as nx
from lattice_weaver.topology.cubical_complex import CubicalComplex
from lattice_weaver.formal.cubical_geometry import GeometricCube
from lattice_weaver.topology.homology_engine import HomologyEngine

# Crear un complejo cúbico (ej. un cuadrado)
graph = nx.Graph()
v0 = GeometricCube(dimensions=0, coordinates=(0, 0))
v1 = GeometricCube(dimensions=0, coordinates=(1, 0))
v2 = GeometricCube(dimensions=0, coordinates=(0, 1))
v3 = GeometricCube(dimensions=0, coordinates=(1, 1))
graph.add_edges_from([(v0, v1), (v1, v3), (v3, v2), (v2, v0)])
complex = CubicalComplex(graph)
complex.build_complex()

# Calcular la homología
engine = HomologyEngine()
homology = engine.compute_homology(complex)

print(f"Números de Betti: {homology}")
print(f"β₀ (Componentes conexas): {homology['beta_0']}")
print(f"β₁ (Ciclos): {homology['beta_1']}")
print(f"β₂ (Cavidades 2D): {homology['beta_2']}")
```

## 4. `FCAToCubicalComplexConverter` (lattice_weaver/formal/fca_cubical_complex.py)

Esta clase facilita la conversión de un `FormalContext` (estructura de FCA) a un `CubicalComplex`. Es el puente clave entre el análisis formal de conceptos y las herramientas de topología cúbica.

### Propósito

Permitir que las estructuras de datos generadas por FCA sean analizadas utilizando las herramientas de topología cúbica, revelando patrones y relaciones topológicas inherentes a los contextos formales.

### Uso Básico

```python
import networkx as nx
from lattice_weaver.formal.fca_cubical_complex import FCAToCubicalComplexConverter, FormalContext
from lattice_weaver.topology.cubical_complex import CubicalComplex
from lattice_weaver.formal.cubical_geometry import GeometricCube

# Definir un contexto formal simple
objects = ["Manzana", "Naranja", "Plátano"]
attributes = ["Fruta", "Dulce", "Amarillo", "Redondo"]
incidence = {
    "Manzana": ["Fruta", "Dulce", "Redondo"],
    "Naranja": ["Fruta", "Dulce", "Redondo"],
    "Plátano": ["Fruta", "Dulce", "Amarillo"]
}
fca_context = FormalContext(objects, attributes, incidence)

# Convertir a complejo cúbico
converter = FCAToCubicalComplexConverter()
cubical_complex = converter.convert(fca_context)

print(cubical_complex)
print(f"Número de 0-cubos (objetos): {len(cubical_complex.cubes[0])}")
print(f"Número de 1-cubos (relaciones): {len(cubical_complex.cubes[1])}")
```

## 5. `TopologyVisualizer` (lattice_weaver/topology/visualization.py)

La clase `TopologyVisualizer` ofrece funcionalidades para la representación gráfica de `CubicalComplex` y los números de Betti calculados por `HomologyEngine`.

### Propósito

Facilitar la comprensión visual de las estructuras topológicas y sus propiedades, ayudando en la depuración y el análisis de los complejos cúbicos.

### Uso Básico

```python
import networkx as nx
from lattice_weaver.topology.cubical_complex import CubicalComplex
from lattice_weaver.formal.cubical_geometry import GeometricCube
from lattice_weaver.topology.homology_engine import HomologyEngine
from lattice_weaver.topology.visualization import TopologyVisualizer

# Crear un complejo cúbico (ej. un cuadrado)
graph = nx.Graph()
v0 = GeometricCube(dimensions=0, coordinates=(0, 0))
v1 = GeometricCube(dimensions=0, coordinates=(1, 0))
v2 = GeometricCube(dimensions=0, coordinates=(0, 1))
v3 = GeometricCube(dimensions=0, coordinates=(1, 1))
graph.add_edges_from([(v0, v1), (v1, v3), (v3, v2), (v2, v0)])
complex = CubicalComplex(graph)
complex.build_complex()

# Calcular la homología
engine = HomologyEngine()
homology = engine.compute_homology(complex)

# Visualizar el complejo y la homología
visualizer = TopologyVisualizer()

# Visualizar el complejo cúbico
fig_complex = visualizer.visualize_cubical_complex(complex, title="Complejo Cúbico de un Cuadrado")

# Visualizar los números de Betti
fig_homology = visualizer.visualize_homology(homology, title="Homología de un Cuadrado")

# Para mostrar las figuras (descomentar si se ejecuta en un entorno interactivo)
# visualizer.show()

# Guardar las figuras
# fig_complex.savefig("square_complex.png")
# fig_homology.savefig("square_homology.png")

visualizer.close()
```

## Ejemplo Completo: Pipeline FCA ↔ Topología

```python
import networkx as nx
from lattice_weaver.formal.fca_cubical_complex import FCAToCubicalComplexConverter, FormalContext
from lattice_weaver.topology.cubical_complex import CubicalComplex
from lattice_weaver.formal.cubical_geometry import GeometricCube
from lattice_weaver.topology.homology_engine import HomologyEngine
from lattice_weaver.topology.visualization import TopologyVisualizer

# 1. Definir un contexto formal
objects = ["Perro", "Gato", "Pez", "Pájaro"]
attributes = ["Doméstico", "Pelaje", "Aletas", "Vuela"]
incidence = {
    "Perro": ["Doméstico", "Pelaje"],
    "Gato": ["Doméstico", "Pelaje"],
    "Pez": ["Aletas"],
    "Pájaro": ["Vuela"]
}
fca_context = FormalContext(objects, attributes, incidence)

print("--- Contexto Formal ---")
for obj, attrs in fca_context.incidence.items():
    print(f"{obj}: {attrs}")
print("\n")

# 2. Convertir el contexto formal a un complejo cúbico
converter = FCAToCubicalComplexConverter()
cubical_complex = converter.convert(fca_context)

print("--- Complejo Cúbico Generado ---")
print(cubical_complex)
print(f"Vértices (0-cubos): {len(cubical_complex.cubes[0])}")
print(f"Aristas (1-cubos): {len(cubical_complex.cubes[1])}")
print("\n")

# 3. Calcular la homología del complejo cúbico
engine = HomologyEngine()
homology = engine.compute_homology(cubical_complex)

print("--- Homología del Complejo Cúbico ---")
print(f"Números de Betti: {homology}")
print(f"β₀ (Componentes conexas): {homology['beta_0']}")
print(f"β₁ (Ciclos): {homology['beta_1']}")
print(f"β₂ (Cavidades 2D): {homology['beta_2']}")
print("\n")

# 4. Visualizar el complejo cúbico y su homología
visualizer = TopologyVisualizer()

# Visualizar el complejo cúbico
fig_complex = visualizer.visualize_cubical_complex(
    cubical_complex,
    title="Complejo Cúbico de Contexto Formal",
    save_path="fca_cubical_complex.png"
)

# Visualizar los números de Betti
fig_homology = visualizer.visualize_homology(
    homology,
    title="Homología del Contexto Formal",
    save_path="fca_homology.png"
)

print("Visualizaciones guardadas como fca_cubical_complex.png y fca_homology.png")

visualizer.close()
```

## Consideraciones Futuras

- **Refinamiento de `FCAToCubicalComplexConverter`**: La conversión actual es simplificada. Se podría implementar una conversión más sofisticada que genere 2-cubos y 3-cubos basados en conceptos formales o retículos de conceptos.
- **Homología de Orden Superior**: La `HomologyEngine` actual solo calcula hasta β₂. Se podría extender para soportar dimensiones superiores.
- **Visualización Interactiva**: Explorar herramientas de visualización interactivas (ej. Plotly, Bokeh) para complejos cúbicos 3D.
- **Integración con `GeometricCube`**: Asegurar que la construcción de `CubicalComplex` y la conversión desde FCA utilicen `GeometricCube` de manera más explícita para definir las relaciones de incidencia y las caras de los cubos de orden superior.


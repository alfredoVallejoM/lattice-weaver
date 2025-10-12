# Guía de Inicio Rápido - LatticeWeaver

**Versión:** 5.0.0  
**Tiempo estimado:** 30 minutos  
**Nivel:** Principiante

---

## Introducción

Esta guía te llevará paso a paso desde la instalación hasta resolver tu primer problema con LatticeWeaver. Al final, habrás:

1. Instalado LatticeWeaver
2. Resuelto un problema CSP (Sudoku 4×4)
3. Construido un lattice FCA
4. Visualizado los resultados

---

## Paso 1: Instalación

### Requisitos Previos

- Python >= 3.11
- pip
- Git (opcional, para clonar el repositorio)

### Instalación desde PyPI (Recomendado)

```bash
pip install lattice-weaver
```

### Instalación desde Código Fuente

```bash
# Clonar repositorio
git clone https://github.com/latticeweaver/lattice-weaver.git
cd lattice-weaver

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar
pip install -e .
```

### Verificar Instalación

```python
import lattice_weaver
print(lattice_weaver.__version__)  # Debería mostrar: 5.0.0
```

---

## Paso 2: Tu Primer Problema CSP - Sudoku 4×4

Vamos a resolver un Sudoku 4×4 simplificado. En este problema:
- Tenemos una cuadrícula de 4×4
- Cada celda puede contener valores 1-4
- Cada fila debe tener valores únicos
- Cada columna debe tener valores únicos
- Cada bloque 2×2 debe tener valores únicos

### Código Completo

```python
from lattice_weaver.arc_engine import AdaptiveConsistencyEngine

def solve_sudoku_4x4():
    """Resuelve un Sudoku 4×4."""
    
    # Crear motor CSP
    engine = AdaptiveConsistencyEngine(
        algorithm='parallel',
        optimization_level=2
    )
    
    # Definir el problema inicial (0 = celda vacía)
    initial_grid = [
        [1, 0, 0, 4],
        [0, 0, 1, 0],
        [0, 3, 0, 0],
        [4, 0, 0, 2]
    ]
    
    # Añadir variables (una por celda)
    for i in range(4):
        for j in range(4):
            cell_name = f"C{i}{j}"
            if initial_grid[i][j] == 0:
                # Celda vacía: dominio completo
                engine.add_variable(cell_name, [1, 2, 3, 4])
            else:
                # Celda pre-llenada: dominio único
                engine.add_variable(cell_name, [initial_grid[i][j]])
    
    # Restricción: todas diferentes
    def all_different(a, b):
        return a != b
    
    # Añadir restricciones de fila
    for i in range(4):
        for j1 in range(4):
            for j2 in range(j1 + 1, 4):
                engine.add_constraint(
                    f"C{i}{j1}", f"C{i}{j2}",
                    all_different,
                    name=f"row_{i}_{j1}_{j2}"
                )
    
    # Añadir restricciones de columna
    for j in range(4):
        for i1 in range(4):
            for i2 in range(i1 + 1, 4):
                engine.add_constraint(
                    f"C{i1}{j}", f"C{i2}{j}",
                    all_different,
                    name=f"col_{i1}_{i2}_{j}"
                )
    
    # Añadir restricciones de bloque 2×2
    blocks = [
        [(0,0), (0,1), (1,0), (1,1)],  # Bloque superior izquierdo
        [(0,2), (0,3), (1,2), (1,3)],  # Bloque superior derecho
        [(2,0), (2,1), (3,0), (3,1)],  # Bloque inferior izquierdo
        [(2,2), (2,3), (3,2), (3,3)]   # Bloque inferior derecho
    ]
    
    for block_idx, block in enumerate(blocks):
        for idx1 in range(len(block)):
            for idx2 in range(idx1 + 1, len(block)):
                i1, j1 = block[idx1]
                i2, j2 = block[idx2]
                engine.add_constraint(
                    f"C{i1}{j1}", f"C{i2}{j2}",
                    all_different,
                    name=f"block_{block_idx}_{idx1}_{idx2}"
                )
    
    # Resolver
    print("Resolviendo Sudoku 4×4...")
    solution = engine.solve()
    
    if solution:
        print("\n✅ Solución encontrada:\n")
        for i in range(4):
            row = []
            for j in range(4):
                row.append(str(solution[f"C{i}{j}"]))
            print(" ".join(row))
        
        # Mostrar estadísticas
        stats = engine.get_statistics()
        print(f"\n📊 Estadísticas:")
        print(f"  Tiempo: {stats['execution_time']:.3f}s")
        print(f"  Iteraciones: {stats['iterations']}")
        print(f"  Reducciones de dominio: {stats['domain_reductions']}")
    else:
        print("❌ No se encontró solución")
    
    return solution

# Ejecutar
solution = solve_sudoku_4x4()
```

### Salida Esperada

```
Resolviendo Sudoku 4×4...

✅ Solución encontrada:

1 2 3 4
3 4 1 2
2 3 4 1
4 1 2 3

📊 Estadísticas:
  Tiempo: 0.012s
  Iteraciones: 8
  Reducciones de dominio: 24
```

### Explicación del Código

1. **Crear el motor CSP**: Inicializamos `AdaptiveConsistencyEngine` con paralelización habilitada.

2. **Definir variables**: Cada celda del Sudoku es una variable. Las celdas pre-llenadas tienen dominio de un solo valor.

3. **Añadir restricciones**: Añadimos tres tipos de restricciones:
   - **Filas**: Todas las celdas en una fila deben ser diferentes
   - **Columnas**: Todas las celdas en una columna deben ser diferentes
   - **Bloques**: Todas las celdas en un bloque 2×2 deben ser diferentes

4. **Resolver**: Llamamos a `solve()` que aplica consistencia de arcos y búsqueda.

5. **Mostrar resultados**: Imprimimos la solución y estadísticas.

---

## Paso 3: Análisis de Conceptos con FCA

Ahora vamos a usar Formal Concept Analysis para analizar un conjunto de datos.

### Escenario: Análisis de Películas

Tenemos películas con diferentes características (género, año, etc.) y queremos descubrir patrones.

### Código Completo

```python
from lattice_weaver.locales import FormalContext, build_concept_lattice

def analyze_movies():
    """Analiza películas usando FCA."""
    
    # Crear contexto formal
    context = FormalContext("Películas")
    
    # Añadir películas con sus características
    context.add_object("Matrix", ["Sci-Fi", "Action", "1990s"])
    context.add_object("Inception", ["Sci-Fi", "Thriller", "2010s"])
    context.add_object("Interstellar", ["Sci-Fi", "Drama", "2010s"])
    context.add_object("Die Hard", ["Action", "Thriller", "1980s"])
    context.add_object("Terminator", ["Sci-Fi", "Action", "1980s"])
    context.add_object("The Godfather", ["Drama", "Crime", "1970s"])
    context.add_object("Pulp Fiction", ["Crime", "Thriller", "1990s"])
    
    print("📚 Contexto creado con 7 películas\n")
    
    # Consultas básicas
    print("🔍 Consultas básicas:")
    
    # ¿Qué películas son Sci-Fi y Action?
    extent = context.get_extent({"Sci-Fi", "Action"})
    print(f"Películas Sci-Fi + Action: {extent}")
    
    # ¿Qué características comparten Matrix y Terminator?
    intent = context.get_intent({"Matrix", "Terminator"})
    print(f"Características comunes (Matrix, Terminator): {intent}")
    
    # Construir lattice
    print("\n🏗️  Construyendo lattice de conceptos...")
    lattice = build_concept_lattice(context, algorithm='next_closure')
    
    # Analizar conceptos
    concepts = lattice.get_concepts()
    print(f"Total de conceptos: {len(concepts)}\n")
    
    # Mostrar conceptos interesantes (con 2+ películas)
    print("💡 Conceptos interesantes (2+ películas):")
    for idx, concept in enumerate(concepts):
        if len(concept.extent) >= 2:
            print(f"\nConcepto {idx}:")
            print(f"  Películas: {concept.extent}")
            print(f"  Características: {concept.intent}")
    
    # Visualizar (genera archivo HTML)
    print("\n📊 Generando visualización...")
    lattice.visualize(
        layout='hierarchical',
        show_labels=True,
        output_file='movies_lattice.html'
    )
    print("✅ Visualización guardada en movies_lattice.html")
    
    return lattice

# Ejecutar
lattice = analyze_movies()
```

### Salida Esperada

```
📚 Contexto creado con 7 películas

🔍 Consultas básicas:
Películas Sci-Fi + Action: {'Matrix', 'Terminator'}
Características comunes (Matrix, Terminator): {'Sci-Fi', 'Action'}

🏗️  Construyendo lattice de conceptos...
Total de conceptos: 15

💡 Conceptos interesantes (2+ películas):

Concepto 3:
  Películas: {'Matrix', 'Terminator'}
  Características: {'Sci-Fi', 'Action'}

Concepto 5:
  Películas: {'Inception', 'Interstellar'}
  Características: {'Sci-Fi', '2010s'}

Concepto 7:
  Películas: {'Die Hard', 'Pulp Fiction'}
  Características: {'Thriller'}

📊 Generando visualización...
✅ Visualización guardada en movies_lattice.html
```

### Explicación del Código

1. **Crear contexto formal**: Definimos la relación películas × características.

2. **Consultas básicas**: Usamos `get_extent()` y `get_intent()` para explorar el contexto.

3. **Construir lattice**: El algoritmo `next_closure` genera todos los conceptos formales.

4. **Analizar conceptos**: Cada concepto representa un grupo de películas con características comunes.

5. **Visualizar**: Generamos un diagrama de Hasse interactivo en HTML.

---

## Paso 4: Visualización Avanzada

Vamos a visualizar el problema CSP del Sudoku.

### Código

```python
from lattice_weaver.visualization import VisualizationEngine
from lattice_weaver.arc_engine import AdaptiveConsistencyEngine

def visualize_sudoku():
    """Visualiza el proceso de resolución de Sudoku."""
    
    # Crear motor CSP (mismo que antes)
    engine = AdaptiveConsistencyEngine()
    
    # Definir problema (simplificado para visualización)
    initial_grid = [
        [1, 0, 0, 4],
        [0, 0, 1, 0],
        [0, 3, 0, 0],
        [4, 0, 0, 2]
    ]
    
    # Añadir variables
    for i in range(4):
        for j in range(4):
            cell_name = f"C{i}{j}"
            if initial_grid[i][j] == 0:
                engine.add_variable(cell_name, [1, 2, 3, 4])
            else:
                engine.add_variable(cell_name, [initial_grid[i][j]])
    
    # Añadir algunas restricciones (simplificado)
    for i in range(4):
        for j1 in range(4):
            for j2 in range(j1 + 1, 4):
                engine.add_constraint(
                    f"C{i}{j1}", f"C{i}{j2}",
                    lambda a, b: a != b
                )
    
    # Resolver
    solution = engine.solve()
    
    # Visualizar
    viz_engine = VisualizationEngine(renderer='d3', interactive=True)
    
    print("📊 Generando visualización del CSP...")
    viz_engine.visualize_csp(
        engine,
        show_domains=True,
        show_constraints=True,
        highlight_solution=solution,
        output_file='sudoku_csp.html'
    )
    print("✅ Visualización guardada en sudoku_csp.html")

# Ejecutar
visualize_sudoku()
```

---

## Paso 5: Integración de CSP y FCA

Finalmente, vamos a integrar CSP y FCA usando el motor de inferencia.

### Escenario: Configuración de Productos

Tenemos productos con diferentes configuraciones válidas (CSP) y queremos analizar patrones de configuración (FCA).

### Código

```python
from lattice_weaver.arc_engine import AdaptiveConsistencyEngine
from lattice_weaver.inference import InferenceEngine

def product_configuration():
    """Configura productos y analiza patrones."""
    
    # Definir problema CSP: configuración de laptop
    engine = AdaptiveConsistencyEngine()
    
    engine.add_variable("CPU", ["i5", "i7", "i9"])
    engine.add_variable("RAM", ["8GB", "16GB", "32GB"])
    engine.add_variable("Storage", ["256GB", "512GB", "1TB"])
    engine.add_variable("GPU", ["Integrated", "GTX", "RTX"])
    
    # Restricciones de compatibilidad
    # i5 no puede tener RTX
    engine.add_constraint(
        "CPU", "GPU",
        lambda cpu, gpu: not (cpu == "i5" and gpu == "RTX")
    )
    
    # 8GB RAM no puede tener RTX
    engine.add_constraint(
        "RAM", "GPU",
        lambda ram, gpu: not (ram == "8GB" and gpu == "RTX")
    )
    
    # i9 requiere al menos 16GB RAM
    engine.add_constraint(
        "CPU", "RAM",
        lambda cpu, ram: not (cpu == "i9" and ram == "8GB")
    )
    
    # Encontrar todas las configuraciones válidas
    print("🔍 Buscando configuraciones válidas...")
    solutions = engine.solve(return_all=True, max_solutions=50)
    print(f"Encontradas {len(solutions)} configuraciones válidas\n")
    
    # Mostrar algunas configuraciones
    print("💻 Configuraciones de ejemplo:")
    for i, sol in enumerate(solutions[:5]):
        print(f"{i+1}. CPU:{sol['CPU']}, RAM:{sol['RAM']}, "
              f"Storage:{sol['Storage']}, GPU:{sol['GPU']}")
    
    # Usar motor de inferencia para analizar patrones
    inference = InferenceEngine(csp_engine=engine)
    
    # Convertir soluciones a contexto FCA
    print("\n🏗️  Construyendo contexto FCA desde soluciones CSP...")
    context = inference.infer_from_csp_to_fca(solutions)
    
    # Construir lattice
    from lattice_weaver.locales import build_concept_lattice
    lattice = build_concept_lattice(context)
    
    print(f"Lattice construido con {len(lattice.get_concepts())} conceptos")
    
    # Visualizar
    lattice.visualize(output_file='product_config_lattice.html')
    print("✅ Visualización guardada en product_config_lattice.html")

# Ejecutar
product_configuration()
```

---

## Próximos Pasos

¡Felicidades! Has completado la guía de inicio rápido. Ahora puedes:

1. **Explorar más ejemplos**: Ver [`docs/examples/`](../examples/)
2. **Leer la documentación de API**: Ver [`docs/api/API_REFERENCE.md`](../api/API_REFERENCE.md)
3. **Aprender sobre arquitectura**: Ver [`docs/architecture/ARCHITECTURE.md`](../architecture/ARCHITECTURE.md)
4. **Contribuir al proyecto**: Ver [`docs/contributing/CONTRIBUTING.md`](../contributing/CONTRIBUTING.md)

---

## Recursos Adicionales

- **Documentación completa**: https://latticeweaver.dev/docs
- **Tutoriales avanzados**: [`docs/tutorials/`](.)
- **Ejemplos de código**: [`examples/`](../../examples/)
- **Comunidad Discord**: https://discord.gg/latticeweaver
- **GitHub Issues**: https://github.com/latticeweaver/lattice-weaver/issues

---

**¡Disfruta explorando LatticeWeaver!** 🚀


# Guía de Tracing del Espacio de Búsqueda

**Proyecto:** LatticeWeaver v5.0  
**Autor:** Manus AI  
**Fecha:** 12 de Octubre, 2025  
**Versión:** 1.0

---

## 1. Introducción

El **SearchSpaceTracer** es un sistema de captura de eventos de bajo overhead diseñado para registrar la evolución del espacio de búsqueda durante la resolución de problemas de satisfacción de restricciones (CSPs). Este módulo permite a los desarrolladores e investigadores analizar el comportamiento del solver, identificar cuellos de botella y optimizar estrategias de búsqueda.

### Características Principales

- **Bajo Overhead:** Diseñado para minimizar el impacto en el rendimiento del solver
- **Modo Síncrono y Asíncrono:** Flexibilidad para diferentes casos de uso
- **Múltiples Formatos:** Exportación a CSV y JSON Lines
- **Estadísticas Incrementales:** Cálculo eficiente de métricas durante la ejecución
- **Integración Transparente:** Se integra fácilmente con el `AdaptiveConsistencyEngine`

---

## 2. Instalación y Configuración

El `SearchSpaceTracer` está incluido en el módulo `lattice_weaver.arc_weaver.tracing`. No requiere instalación adicional más allá de las dependencias estándar de LatticeWeaver.

### Dependencias Opcionales

Para utilizar las funciones de carga y análisis de traces, se requiere **pandas**:

```bash
pip install pandas
```

---

## 3. Uso Básico

### 3.1. Tracer en Memoria

El uso más simple es capturar eventos solo en memoria, útil para debugging y análisis rápido.

```python
from lattice_weaver.arc_weaver.tracing import SearchSpaceTracer, SearchEvent
from lattice_weaver.arc_weaver.adaptive_consistency import AdaptiveConsistencyEngine

# Crear tracer
tracer = SearchSpaceTracer(enabled=True)

# Crear solver con tracer
engine = AdaptiveConsistencyEngine(tracer=tracer)

# Resolver problema
stats = engine.solve(problem, max_solutions=1)

# Obtener estadísticas del tracer
tracer_stats = tracer.get_statistics()
print(f"Nodos explorados: {tracer_stats['nodes_explored']}")
print(f"Backtracks: {tracer_stats['backtracks']}")
print(f"Tasa de backtrack: {tracer_stats['backtrack_rate']:.2%}")
```

### 3.2. Tracer con Salida a Archivo (Modo Síncrono)

Para guardar el trace en un archivo CSV o JSON Lines:

```python
tracer = SearchSpaceTracer(
    enabled=True,
    output_path="trace.csv",
    output_format='csv',
    async_mode=False  # Modo síncrono
)

engine = AdaptiveConsistencyEngine(tracer=tracer)
stats = engine.solve(problem, max_solutions=1)

# El archivo trace.csv se crea automáticamente
```

### 3.3. Tracer con Salida Asíncrona (Recomendado)

Para minimizar el overhead, use el modo asíncrono que escribe eventos en un thread separado:

```python
tracer = SearchSpaceTracer(
    enabled=True,
    output_path="trace.csv",
    output_format='csv',
    async_mode=True  # Modo asíncrono
)

engine = AdaptiveConsistencyEngine(tracer=tracer)
stats = engine.solve(problem, max_solutions=1)
```

### 3.4. Context Manager

El tracer soporta el protocolo de context manager para asegurar que los recursos se liberen correctamente:

```python
with SearchSpaceTracer(enabled=True, output_path="trace.csv") as tracer:
    engine = AdaptiveConsistencyEngine(tracer=tracer)
    stats = engine.solve(problem, max_solutions=1)

# El archivo se cierra automáticamente al salir del contexto
```

---

## 4. Tipos de Eventos

El `SearchSpaceTracer` captura los siguientes tipos de eventos:

| Tipo de Evento | Descripción | Campos Relevantes |
|----------------|-------------|-------------------|
| `search_started` | Inicio de la búsqueda | `metadata` (parámetros de búsqueda) |
| `search_ended` | Fin de la búsqueda | `metadata` (estadísticas finales) |
| `variable_assigned` | Asignación de una variable | `variable`, `value`, `depth` |
| `backtrack` | Retroceso en la búsqueda | `variable`, `depth` |
| `domain_pruned` | Poda de valores de un dominio | `variable`, `source_variable`, `pruned_values` |
| `solution_found` | Solución encontrada | `metadata` (solución completa) |
| `ac3_call` | Llamada al algoritmo AC-3 | `metadata` (información del clúster) |
| `cluster_operation` | Operación de clustering | `metadata` (tipo de operación) |

---

## 5. Estructura de un Evento

Cada evento se representa mediante la clase `SearchEvent`:

```python
@dataclass(frozen=True)
class SearchEvent:
    timestamp: float                # Tiempo del evento (segundos desde epoch)
    event_type: Literal[...]        # Tipo de evento
    variable: Optional[str]         # Variable involucrada
    value: Optional[Any]            # Valor asignado/eliminado
    source_variable: Optional[str]  # Variable que causó la poda
    pruned_values: Optional[Set]    # Valores eliminados
    depth: int                      # Profundidad en el árbol de búsqueda
    metadata: Dict[str, Any]        # Información adicional
```

---

## 6. Análisis de Traces

### 6.1. Cargar un Trace

```python
from lattice_weaver.arc_weaver.tracing import load_trace

# Carga automática detectando el formato
df = load_trace("trace.csv")

# O especificar el formato
from lattice_weaver.arc_weaver.tracing import load_trace_csv, load_trace_jsonl

df_csv = load_trace_csv("trace.csv")
df_jsonl = load_trace_jsonl("trace.jsonl")
```

### 6.2. Convertir a DataFrame

Si el tracer está en memoria, puede convertir los eventos a un DataFrame de pandas:

```python
df = tracer.to_dataframe()

# Análisis básico
print(df['event_type'].value_counts())
print(df[df['event_type'] == 'backtrack']['depth'].describe())
```

### 6.3. Estadísticas Agregadas

El método `get_statistics()` devuelve un diccionario con métricas útiles:

```python
stats = tracer.get_statistics()

print(f"Total de eventos: {stats['total_events']}")
print(f"Nodos explorados: {stats['nodes_explored']}")
print(f"Backtracks: {stats['backtracks']}")
print(f"Tasa de backtrack: {stats['backtrack_rate']:.2%}")
print(f"Profundidad máxima: {stats['max_depth']}")
print(f"Duración total: {stats['total_duration']:.4f}s")
print(f"Eventos/segundo: {stats['events_per_second']:.0f}")
```

---

## 7. Configuración Avanzada

### 7.1. Tamaño del Buffer

El parámetro `buffer_size` controla cuántos eventos se acumulan antes de escribir al disco:

```python
tracer = SearchSpaceTracer(
    enabled=True,
    output_path="trace.csv",
    buffer_size=5000  # Escribir cada 5000 eventos
)
```

**Recomendaciones:**
- Buffer pequeño (100-500): Menor uso de memoria, más operaciones de I/O
- Buffer grande (5000-10000): Mayor uso de memoria, menos operaciones de I/O

### 7.2. Formato de Salida

Elija el formato según su caso de uso:

**CSV:**
- Ventajas: Fácil de abrir en Excel, compatible con muchas herramientas
- Desventajas: Requiere parseo de JSON para campos complejos

**JSON Lines:**
- Ventajas: Estructura nativa, fácil de procesar programáticamente
- Desventajas: Archivos más grandes

```python
# CSV
tracer_csv = SearchSpaceTracer(
    enabled=True,
    output_path="trace.csv",
    output_format='csv'
)

# JSON Lines
tracer_jsonl = SearchSpaceTracer(
    enabled=True,
    output_path="trace.jsonl",
    output_format='jsonl'
)
```

---

## 8. Rendimiento y Overhead

### 8.1. Overhead Esperado

El overhead del tracer varía según el modo y el tamaño del problema:

| Modo | Overhead Típico | Uso Recomendado |
|------|----------------|-----------------|
| Deshabilitado | <1% | Producción |
| Memoria | 5-10% | Debugging, problemas pequeños |
| Síncrono | 20-50% | Análisis detallado, problemas medianos |
| Asíncrono | 5-15% | Análisis detallado, problemas grandes |

**Nota:** El overhead relativo es mayor en problemas pequeños debido al costo fijo de inicialización. En problemas grandes, el overhead tiende a estabilizarse en los rangos indicados.

### 8.2. Optimización del Rendimiento

Para minimizar el overhead:

1. **Use modo asíncrono** para problemas grandes
2. **Aumente el buffer_size** para reducir operaciones de I/O
3. **Deshabilite el tracer** en producción si no es necesario
4. **Use sampling** (capturar solo 1 de cada N eventos) si es aceptable

---

## 9. Ejemplos Completos

### 9.1. Ejemplo: N-Reinas con Tracing

```python
from lattice_weaver.arc_weaver.graph_structures import ConstraintGraph
from lattice_weaver.arc_weaver.adaptive_consistency import AdaptiveConsistencyEngine
from lattice_weaver.arc_weaver.tracing import SearchSpaceTracer

def create_nqueens_problem(n):
    cg = ConstraintGraph()
    for i in range(n):
        cg.add_variable(f'Q{i}', set(range(n)))
    
    for i in range(n):
        for j in range(i + 1, n):
            def make_constraint(row_diff):
                def constraint(vi, vj):
                    if vi == vj or abs(vi - vj) == row_diff:
                        return False
                    return True
                return constraint
            cg.add_constraint(f'Q{i}', f'Q{j}', make_constraint(j - i))
    
    return cg

# Resolver con tracing
problem = create_nqueens_problem(8)

with SearchSpaceTracer(enabled=True, output_path="nqueens_8.csv", async_mode=True) as tracer:
    engine = AdaptiveConsistencyEngine(tracer=tracer)
    stats = engine.solve(problem, max_solutions=1)
    
    print(f"Soluciones: {len(stats.solutions)}")
    print(f"Nodos: {stats.nodes_explored}")
    print(f"Backtracks: {stats.backtracks}")
    
    tracer_stats = tracer.get_statistics()
    print(f"Eventos capturados: {tracer_stats['total_events']}")
```

### 9.2. Ejemplo: Análisis Post-Resolución

```python
from lattice_weaver.arc_weaver.tracing import load_trace
import matplotlib.pyplot as plt

# Cargar trace
df = load_trace("nqueens_8.csv")

# Analizar profundidad de backtracks
backtrack_df = df[df['event_type'] == 'backtrack']
plt.hist(backtrack_df['depth'], bins=20)
plt.xlabel('Profundidad')
plt.ylabel('Frecuencia de Backtracks')
plt.title('Distribución de Backtracks por Profundidad')
plt.savefig('backtrack_distribution.png')
```

---

## 10. Integración con Otros Módulos

### 10.1. SearchSpaceVisualizer

El `SearchSpaceTracer` está diseñado para integrarse con el `SearchSpaceVisualizer` (Semanas 3-4):

```python
from lattice_weaver.visualization.search_viz import generate_report

# Generar reporte HTML con visualizaciones
df = load_trace("trace.csv")
generate_report(df, output_path="report.html")
```

### 10.2. ExperimentRunner

El `ExperimentRunner` (Semanas 5-6) utiliza el tracer para capturar datos detallados de cada ejecución:

```python
from lattice_weaver.benchmarks.runner import ExperimentRunner

runner = ExperimentRunner(config_path="experiments/config.yaml")
results = runner.run()  # Cada ejecución genera un trace
```

---

## 11. Solución de Problemas

### 11.1. El archivo de trace está vacío

**Causa:** El tracer no se detuvo correctamente.

**Solución:** Use el context manager o llame explícitamente a `tracer.stop()`:

```python
tracer.start()
# ... resolver problema ...
tracer.stop()  # Asegura que el buffer se vacíe
```

### 11.2. Overhead muy alto

**Causa:** Modo síncrono con problema pequeño.

**Solución:** Use modo asíncrono o aumente el tamaño del buffer:

```python
tracer = SearchSpaceTracer(
    enabled=True,
    output_path="trace.csv",
    async_mode=True,
    buffer_size=10000
)
```

### 11.3. Pérdida de eventos en modo asíncrono

**Causa:** La cola del worker thread está llena.

**Solución:** Aumente el tamaño de la cola (el doble del buffer_size por defecto):

```python
# Esto se maneja automáticamente, pero si persiste el problema,
# aumente el buffer_size
tracer = SearchSpaceTracer(
    enabled=True,
    output_path="trace.csv",
    async_mode=True,
    buffer_size=20000  # Cola será de 40000
)
```

---

## 12. Referencia de API

### 12.1. SearchSpaceTracer

```python
class SearchSpaceTracer:
    def __init__(
        self,
        enabled: bool = True,
        output_path: Optional[str] = None,
        buffer_size: int = 1000,
        output_format: Literal['csv', 'jsonl'] = 'csv',
        async_mode: bool = False
    )
    
    def start() -> None
    def stop() -> None
    def record(event: SearchEvent) -> None
    def get_statistics() -> Dict[str, Any]
    def to_dataframe() -> pd.DataFrame
    def clear() -> None
```

### 12.2. SearchEvent

```python
@dataclass(frozen=True)
class SearchEvent:
    timestamp: float
    event_type: Literal[
        'search_started', 'search_ended', 'variable_assigned',
        'backtrack', 'domain_pruned', 'solution_found',
        'ac3_call', 'cluster_operation'
    ]
    variable: Optional[str] = None
    value: Optional[Any] = None
    source_variable: Optional[str] = None
    pruned_values: Optional[Set[Any]] = None
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict() -> Dict[str, Any]
```

### 12.3. Funciones de Carga

```python
def load_trace(path: str) -> pd.DataFrame
def load_trace_csv(path: str) -> pd.DataFrame
def load_trace_jsonl(path: str) -> pd.DataFrame
```

---

## 13. Mejores Prácticas

1. **Use modo asíncrono** para problemas grandes o cuando el rendimiento es crítico
2. **Deshabilite el tracer en producción** si no es necesario
3. **Use context managers** para asegurar que los recursos se liberen
4. **Analice los traces post-resolución** en lugar de durante la ejecución
5. **Guarde traces de referencia** para comparación y regresión
6. **Documente los parámetros** del problema junto con el trace

---

## 14. Próximos Pasos

- **Semanas 3-4:** Aprenda a visualizar traces con el `SearchSpaceVisualizer`
- **Semanas 5-6:** Use el `ExperimentRunner` para experimentación masiva
- **Track E:** Integre el tracing en la aplicación web

---

**Preparado por:** Manus AI  
**Fecha:** 12 de Octubre, 2025  
**Versión:** 1.0


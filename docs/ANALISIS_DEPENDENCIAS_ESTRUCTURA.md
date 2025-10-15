# Análisis de Dependencias y Estructura Global - Lattice Weaver

**Versión**: 1.0  
**Fecha**: 15 de Octubre, 2025  
**Autor**: Agente Autónomo - Lattice Weaver

---

## Introducción

Este documento proporciona un análisis exhaustivo de las dependencias entre los distintos módulos de Lattice Weaver y detalla la estructura global de la librería, incluyendo el estado actual y la visión futura. Este análisis es esencial para entender cómo los diferentes componentes interactúan y para planificar el desarrollo futuro de manera coherente.

## Estructura de Módulos

### Vista General

```
lattice-weaver/
├── lattice_weaver/
│   ├── __init__.py
│   ├── arc_engine/          # Motor de consistencia de arcos
│   ├── fibration/           # Core de Fibration Flow
│   ├── homotopy/            # Reglas de homotopía
│   ├── utils/               # Utilidades y optimizaciones
│   └── [otros módulos...]
├── tests/
│   ├── unit/                # Tests unitarios
│   └── integration/         # Tests de integración
├── benchmarks/              # Benchmarks de rendimiento
└── docs/                    # Documentación
```

### Módulos Core

#### 1. `arc_engine/` - Motor de Consistencia de Arcos

**Propósito**: Implementa algoritmos de propagación de restricciones (AC-3, Forward Checking, etc.) y gestión de dominios.

**Archivos Principales**:
- `core.py`: Clase principal `ArcEngine`
- `domains.py`: Representación de dominios (SetDomain, RangeDomain)
- `tms.py`: Truth Maintenance System para backtracking
- `advanced_optimizations.py`: Optimizaciones avanzadas (memoization, spatial index)
- `adaptive_propagation.py`: Propagación adaptativa (Fase 2)
- `tms_enhanced.py`: TMS mejorado con CBJ y no-good learning (Fase 2)

**Dependencias**:
- Depende de: `utils/` (para estructuras de datos)
- Usado por: `fibration/` (HacificationEngine, FibrationSearchSolver)

**Estado Actual**: Maduro y estable. Las optimizaciones de Fase 2 añaden capacidades avanzadas sin romper la API existente.

**Evolución Futura**: Integración más profunda con Sparse Set para representación de dominios. Paralelización de propagación para problemas grandes.

#### 2. `fibration/` - Core de Fibration Flow

**Propósito**: Implementa el algoritmo de Fibration Flow para resolución de problemas CSP con restricciones jerárquicas y optimización.

**Archivos Principales**:
- `constraint_hierarchy.py`: Jerarquía de restricciones (LOCAL, PATTERN, GLOBAL)
- `hacification_engine_optimized.py`: Motor de hacificación optimizado (Fase 1)
- `energy_landscape_optimized.py`: Cálculo de energía optimizado
- `energy_landscape_lazy.py`: Cálculo de energía lazy (Fase 2)
- `fibration_search_solver.py`: Solver de búsqueda original
- `fibration_search_solver_enhanced.py`: Solver mejorado (Fase 1)
- `fibration_search_solver_adaptive.py`: Solver adaptativo (Fase 1)
- `fibration_search_solver_adaptive_v2.py`: Solver adaptativo v2 (Fase 1)
- `hacification_incremental.py`: Hacificación incremental (Fase 3)
- `watched_literals.py`: Watched literals para restricciones (Fase 2)
- `advanced_heuristics.py`: Heurísticas avanzadas (WDeg, IBS, CDVO) (Fase 2)
- `predicate_cache.py`: Cache de predicados (Fase 2)
- `global_constraints.py`: Restricciones globales especializadas (Fase 1)
- `hybrid_search.py`: Búsqueda híbrida (Fase 1)
- `general_constraint.py`: Clase de restricción general (Fase 2)

**Dependencias**:
- Depende de: `arc_engine/`, `homotopy/`, `utils/`
- Usado por: Aplicaciones externas, benchmarks

**Estado Actual**: En desarrollo activo. Las optimizaciones de Fases 1-3 añaden múltiples versiones mejoradas de componentes core.

**Evolución Futura**: Consolidación de las múltiples versiones de solvers en una única implementación con switches configurables. Integración completa de todas las optimizaciones.

#### 3. `homotopy/` - Reglas de Homotopía

**Propósito**: Implementa reglas de homotopía para análisis de dependencias entre restricciones y guía de búsqueda.

**Archivos Principales**:
- `rules.py`: Clase principal `HomotopyRules`
- `rules_optimized.py`: HomotopyRules optimizado con sparse graph (Fase 2)

**Dependencias**:
- Depende de: `fibration/` (para acceso a restricciones)
- Usado por: `fibration/` (FibrationSearchSolver)

**Estado Actual**: Estable. La versión optimizada de Fase 2 mejora significativamente la escalabilidad.

**Evolución Futura**: Lazy computation más agresiva. Caching de resultados entre llamadas.

#### 4. `utils/` - Utilidades y Optimizaciones

**Propósito**: Proporciona utilidades genéricas y optimizaciones de rendimiento reutilizables.

**Archivos Principales**:
- `sparse_set.py`: Estructura de datos Sparse Set (Fase 1)
- `object_pool.py`: Object pooling genérico (Fase 1)
- `auto_profiler.py`: Profiling automático (Fase 1)
- `lazy_init.py`: Lazy initialization (Fase 1)
- `jit_compiler.py`: Compilación JIT con Numba (Fase 3)
- `numpy_vectorization.py`: Vectorización con NumPy (Fase 3)

**Dependencias**:
- Depende de: Bibliotecas estándar, NumPy, Numba
- Usado por: Todos los módulos

**Estado Actual**: Nuevo. Implementado en Fases 1-3 como parte de las optimizaciones de rendimiento.

**Evolución Futura**: Expansión con más utilidades según necesidades. Integración más profunda con componentes core.

## Grafo de Dependencias

### Dependencias entre Módulos

```
utils/
  ↑
  ├── arc_engine/
  │     ↑
  │     └── fibration/
  │           ↑
  │           └── homotopy/
  │
  └── homotopy/
```

### Dependencias Detalladas

#### Nivel 1: Utilidades Base
- `utils/sparse_set.py`: Sin dependencias internas
- `utils/object_pool.py`: Sin dependencias internas
- `utils/auto_profiler.py`: Depende de `psutil`
- `utils/lazy_init.py`: Sin dependencias internas
- `utils/jit_compiler.py`: Depende de `numba`, `numpy`
- `utils/numpy_vectorization.py`: Depende de `numpy`

#### Nivel 2: Arc Engine
- `arc_engine/core.py`: Depende de `domains.py`, `tms.py`
- `arc_engine/domains.py`: Puede usar `utils/sparse_set.py` (futuro)
- `arc_engine/tms.py`: Sin dependencias internas significativas
- `arc_engine/adaptive_propagation.py`: Depende de `core.py`
- `arc_engine/tms_enhanced.py`: Depende de `tms.py`

#### Nivel 3: Homotopy
- `homotopy/rules.py`: Depende de `fibration/constraint_hierarchy.py`
- `homotopy/rules_optimized.py`: Depende de `rules.py`

#### Nivel 4: Fibration (Core)
- `fibration/constraint_hierarchy.py`: Sin dependencias internas significativas
- `fibration/energy_landscape_optimized.py`: Depende de `constraint_hierarchy.py`
- `fibration/energy_landscape_lazy.py`: Depende de `energy_landscape_optimized.py`, `utils/lazy_init.py`
- `fibration/hacification_engine_optimized.py`: Depende de `constraint_hierarchy.py`, `energy_landscape_optimized.py`, `arc_engine/core.py`
- `fibration/hacification_incremental.py`: Depende de `hacification_engine_optimized.py`

#### Nivel 5: Fibration (Solver)
- `fibration/fibration_search_solver.py`: Depende de `hacification_engine_optimized.py`, `homotopy/rules.py`
- `fibration/fibration_search_solver_enhanced.py`: Depende de `fibration_search_solver.py`, `arc_engine/tms_enhanced.py`, `homotopy/rules_optimized.py`
- `fibration/fibration_search_solver_adaptive.py`: Depende de `fibration_search_solver_enhanced.py`, `utils/auto_profiler.py`
- `fibration/fibration_search_solver_adaptive_v2.py`: Depende de `fibration_search_solver_adaptive.py`

#### Nivel 6: Fibration (Optimizaciones)
- `fibration/watched_literals.py`: Depende de `general_constraint.py`
- `fibration/advanced_heuristics.py`: Depende de `general_constraint.py`, `utils/jit_compiler.py`
- `fibration/predicate_cache.py`: Depende de `general_constraint.py`
- `fibration/global_constraints.py`: Depende de `constraint_hierarchy.py`
- `fibration/hybrid_search.py`: Depende de `fibration_search_solver.py`

## Análisis de Acoplamiento

### Acoplamiento Fuerte (Problemático)

**Problema**: Múltiples versiones de solvers (`fibration_search_solver.py`, `*_enhanced.py`, `*_adaptive.py`, `*_adaptive_v2.py`) crean confusión sobre cuál usar.

**Solución Propuesta**: Consolidar en una única implementación con switches configurables y detección automática vía AutoProfiler.

### Acoplamiento Medio (Aceptable)

**HacificationEngine ↔ ArcEngine**: El HacificationEngine depende fuertemente del ArcEngine para propagación. Esto es aceptable dado que son responsabilidades claramente separadas.

**FibrationSearchSolver ↔ HomotopyRules**: El solver usa HomotopyRules para heurísticas. Esto es aceptable pero podría hacerse más flexible con dependency injection.

### Acoplamiento Bajo (Ideal)

**utils/ ↔ Otros módulos**: Las utilidades son genéricas y no dependen de detalles específicos de otros módulos. Esto es ideal y facilita reutilización.

## Estado Actual vs Futuro

### Estado Actual

**Fortalezas**:
- Separación clara de responsabilidades entre módulos
- Utilidades genéricas reutilizables
- Múltiples opciones de optimización disponibles
- Tests exhaustivos (93.7% cobertura)

**Debilidades**:
- Múltiples versiones de componentes similares (solvers)
- Integración incompleta de optimizaciones
- Overhead en problemas pequeños
- Falta de configuración centralizada

### Visión Futura

#### Fase de Consolidación (1-2 meses)

**Objetivos**:
1. Consolidar múltiples versiones de solvers en implementación única
2. Integrar JIT Compiler en operaciones críticas
3. Implementar configuración centralizada
4. Crear documentación de usuario completa

**Resultado Esperado**: Sistema unificado con optimizaciones integradas y configuración simple.

#### Fase de Optimización (2-3 meses)

**Objetivos**:
1. Integrar Sparse Set nativamente en ArcEngine
2. Refactorizar representación de dominios para NumPy
3. Implementar paralelización con Ray
4. Optimizar uso de memoria

**Resultado Esperado**: Speedup de 50-200x en problemas grandes, uso de memoria reducido 10-50x.

#### Fase de Expansión (3-6 meses)

**Objetivos**:
1. Implementar más restricciones globales especializadas
2. Añadir soporte para problemas dinámicos
3. Crear interfaz de alto nivel para usuarios
4. Publicar paper académico

**Resultado Esperado**: Sistema completo y competitivo con estado del arte, con nicho claro en problemas con restricciones SOFT+HARD.

## Recomendaciones de Desarrollo

### Para Nuevas Funcionalidades

1. **Identificar el módulo apropiado**: Determinar dónde encaja naturalmente la funcionalidad
2. **Verificar dependencias**: Asegurar que las dependencias son apropiadas y no crean ciclos
3. **Usar abstracciones existentes**: Reutilizar interfaces y patrones establecidos
4. **Documentar dependencias**: Actualizar este documento cuando se añadan nuevas dependencias
5. **Implementar tests**: Asegurar que la nueva funcionalidad está bien testeada

### Para Refactoring

1. **Analizar impacto**: Usar este documento para identificar qué componentes se verán afectados
2. **Mantener compatibilidad**: Preservar APIs públicas o proveer deprecation warnings
3. **Actualizar tests**: Asegurar que los tests reflejan los cambios
4. **Documentar cambios**: Actualizar documentación y este análisis de dependencias
5. **Validar rendimiento**: Ejecutar benchmarks para asegurar que no hay regresiones

### Para Optimizaciones

1. **Profiling primero**: Identificar cuellos de botella reales
2. **Validar con benchmarks**: Medir impacto en escenarios realistas
3. **Considerar trade-offs**: Evaluar complejidad vs beneficio
4. **Documentar decisiones**: Explicar por qué se eligió una optimización específica
5. **Mantener fallback**: Preservar implementación original para comparación

## Conclusión

La estructura actual de Lattice Weaver es sólida con separación clara de responsabilidades y bajo acoplamiento entre módulos. Las optimizaciones implementadas en Fases 1-3 añaden capacidades significativas sin comprometer la arquitectura base.

El trabajo futuro debe enfocarse en consolidar las múltiples versiones de componentes similares, integrar profundamente las optimizaciones más efectivas (especialmente JIT Compiler), y crear una configuración centralizada que facilite el uso del sistema.

Este documento debe mantenerse actualizado a medida que el proyecto evoluciona, sirviendo como referencia central para entender la estructura y dependencias del sistema.

---

**Versión**: 1.0  
**Última Actualización**: 15 de Octubre, 2025  
**Mantenido por**: Equipo de Desarrollo de Lattice Weaver


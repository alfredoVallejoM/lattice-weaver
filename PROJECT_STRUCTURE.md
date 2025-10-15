# Estructura del Proyecto LatticeWeaver v8.0

**Versión:** 8.0-alpha  
**Fecha:** 15 de Octubre, 2025  
**Arquitectura:** Modular, extensible, compatible por diseño

---

## Estructura de Directorios

```
lattice-weaver/
├── .archive/                           # Archivos obsoletos archivados
│   ├── old-tracks/                     # Directorios de tracks antiguos
│   └── old-docs/                       # Documentación obsoleta
│
├── docs/                               # Documentación principal
│   ├── ARQUITECTURA_COMPLETA_LATTICEWEAVER_V8.md    # Blueprint completo
│   ├── ARQUITECTURA_MODULAR_COMPATIBLE.md           # Arquitectura modular
│   ├── INTEGRACION_COMPLETA_TODAS_CAPACIDADES.md    # Integración funcional
│   ├── PLAN_DE_INTEGRACION_FUNCIONAL.md             # Plan de integración
│   ├── ML_VISION.md                                 # Visión de Mini-IAs
│   ├── ROADMAP_LARGO_PLAZO.md                       # Roadmap 24 meses
│   ├── README_TRACK_A.md                            # Track A (ACE)
│   ├── TRACK_D_INFERENCE_ENGINE_DESIGN.md           # Track D diseño
│   └── ...
│
├── lattice_weaver/                     # Código fuente principal
│   │
│   ├── core/                           # CAPA 1: NÚCLEO
│   │   ├── csp_problem.py              # Definición de CSP
│   │   ├── csp_engine/                 # Motores de resolución
│   │   │   ├── solver.py               # Solver base
│   │   │   └── adaptive_solver.py      # Solver adaptativo
│   │   └── orchestrator.py             # SolverOrchestrator (CAPA 3)
│   │
│   ├── strategies/                     # CAPA 2: ESTRATEGIAS
│   │   ├── base.py                     # Interfaces base
│   │   ├── __init__.py
│   │   ├── analysis/                   # Estrategias de análisis
│   │   │   ├── topological.py          # Track B (Locales)
│   │   │   ├── tda.py                  # TDA
│   │   │   ├── symbolic.py             # Track A (Simbólico)
│   │   │   ├── fca.py                  # FCA
│   │   │   └── symmetry.py             # Formal (Simetrías)
│   │   ├── heuristics/                 # Estrategias heurísticas
│   │   │   ├── family_based.py         # Track C (Familias)
│   │   │   ├── ml_guided.py            # Mini-IAs
│   │   │   ├── mrv.py                  # MRV
│   │   │   ├── degree.py               # Degree
│   │   │   └── lcv.py                  # LCV
│   │   ├── propagation/                # Estrategias de propagación
│   │   │   ├── ace.py                  # Track A (ACE)
│   │   │   ├── modal.py                # Track B (Modal)
│   │   │   └── forward_checking.py     # Forward Checking
│   │   ├── verification/               # Estrategias de verificación
│   │   │   ├── cubical.py              # Tipos Cúbicos
│   │   │   ├── heyting.py              # Lógica Intuicionista
│   │   │   └── property_based.py       # Basada en propiedades
│   │   └── optimization/               # Estrategias de optimización
│   │       ├── symmetry_breaking.py    # Romper simetrías
│   │       ├── dominance.py            # Detectar dominancia
│   │       └── redundancy.py           # Eliminar redundancia
│   │
│   ├── arc_engine/                     # TRACK A: Core Engine (ACE)
│   │   ├── core.py                     # ArcEngine principal
│   │   ├── ac31.py                     # AC-3.1
│   │   ├── optimizations/              # Optimizaciones
│   │   │   ├── symbolic_engine.py      # Motor simbólico
│   │   │   └── speculative_execution.py # Ejecución especulativa
│   │   ├── search_space_tracer.py      # Trazador de búsqueda
│   │   └── experiment_runner.py        # Framework de experimentos
│   │
│   ├── formal/                         # TIPOS CÚBICOS & HoTT
│   │   ├── cubical_syntax.py           # Sintaxis cúbica
│   │   ├── cubical_operations.py       # Operaciones cúbicas
│   │   ├── cubical_engine.py           # Motor de type checking
│   │   ├── csp_cubical_bridge.py       # Gap 1: CSP-Cubical
│   │   ├── cubical_csp_type.py         # CSP como tipo cúbico
│   │   ├── symmetry_extractor.py       # Extracción de simetrías
│   │   ├── path_finder.py              # Búsqueda de caminos
│   │   ├── heyting_algebra.py          # Álgebra de Heyting
│   │   └── type_checker.py             # Type checker
│   │
│   ├── topology/                       # TOPOLOGÍA & TDA
│   │   ├── tda_engine.py               # Motor de TDA
│   │   ├── analyzer.py                 # Análisis topológico
│   │   ├── cubical_complex.py          # Gap 2: Complejos cúbicos
│   │   ├── simplicial_complex.py       # Complejos simpliciales
│   │   └── homology_engine.py          # Cálculo de homología
│   │
│   ├── topology_new/                   # TRACK B: Locales y Frames
│   │   ├── locale.py                   # Locales
│   │   ├── morphisms.py                # Morfismos
│   │   ├── operations.py               # Operaciones modales
│   │   └── ace_bridge.py               # Puente con ACE
│   │
│   ├── lattice_core/                   # FCA (Formal Concept Analysis)
│   │   ├── builder.py                  # Construcción de lattices
│   │   ├── concept.py                  # Conceptos formales
│   │   ├── implications.py             # Implicaciones
│   │   ├── parallel_fca.py             # FCA paralelo
│   │   └── fca_to_cubical.py           # Gap 2: FCA → Cubical
│   │
│   ├── homotopy/                       # HOMOTOPÍA
│   │   ├── analyzer.py                 # Análisis homotópico
│   │   ├── rules.py                    # Reglas de homotopía
│   │   └── verification_bridge.py      # Gap 3: Homotopía-Verificación
│   │
│   ├── problems/                       # TRACK C: Problem Families
│   │   ├── catalog.py                  # Catálogo de problemas
│   │   ├── base.py                     # Clase base
│   │   └── generators/                 # Generadores
│   │       ├── nqueens.py              # N-Queens
│   │       ├── graph_coloring.py       # Graph Coloring
│   │       ├── sudoku.py               # Sudoku
│   │       └── ...                     # 9 familias total
│   │
│   ├── abstraction/                    # ABSTRACCIÓN MULTIESCALA
│   │   └── manager.py                  # AbstractionLevelManager
│   │
│   ├── compiler_multiescala/           # COMPILADOR MULTIESCALA
│   │   ├── base.py                     # AbstractionLevel
│   │   ├── level_0.py                  # Primitivas CSP
│   │   ├── level_1.py                  # Patrones locales
│   │   ├── level_2.py                  # Clusters
│   │   ├── level_3.py                  # Componentes
│   │   ├── level_4.py                  # Simetrías
│   │   ├── level_5.py                  # Estructura algebraica
│   │   └── level_6.py                  # Teoría de categorías
│   │
│   ├── renormalization/                # RENORMALIZACIÓN
│   │   ├── core.py                     # Flujo principal
│   │   ├── partition.py                # Particionamiento
│   │   ├── effective_domains.py        # Dominios efectivos
│   │   └── effective_constraints.py    # Restricciones efectivas
│   │
│   ├── inference/                      # TRACK D: Inference Engine
│   │   ├── parsers/                    # Parsers
│   │   │   ├── json_parser.py          # JSON/YAML
│   │   │   ├── natural_language_parser.py  # Lenguaje natural
│   │   │   └── formal_parser.py        # Especificaciones formales
│   │   ├── ir/                         # Representación intermedia
│   │   │   └── intermediate_representation.py
│   │   ├── inference_layer/            # Capa de inferencia
│   │   │   └── constraint_inferencer.py
│   │   ├── builders/                   # Constructores
│   │   │   └── csp_builder.py
│   │   └── engine.py                   # InferenceEngine
│   │
│   ├── ml/                             # CAPA 5: Mini-IAs (66 modelos)
│   │   ├── accelerator.py              # MLAccelerator
│   │   ├── config.py                   # MLConfig
│   │   └── mini_nets/                  # Suites de Mini-IAs
│   │       ├── arc_engine_suite.py     # 7 Mini-IAs para ACE
│   │       ├── cubical_suite.py        # 10 Mini-IAs para Cubical
│   │       ├── tda_suite.py            # 9 Mini-IAs para TDA
│   │       ├── compiler_suite.py       # 8 Mini-IAs para Compiler
│   │       ├── inference_suite.py      # 7 Mini-IAs para Inference
│   │       ├── fca_suite.py            # 8 Mini-IAs para FCA
│   │       ├── homotopy_suite.py       # 6 Mini-IAs para Homotopía
│   │       ├── meta_suite.py           # 5 Mini-IAs para Meta
│   │       └── renormalization_suite.py # 6 Mini-IAs para Renorm
│   │
│   ├── web/                            # TRACK E: Web Application
│   │   ├── backend/                    # Backend FastAPI
│   │   │   ├── api.py                  # API REST
│   │   │   └── websockets.py           # WebSockets
│   │   └── frontend/                   # Frontend React
│   │       └── (React app)
│   │
│   ├── desktop/                        # TRACK F: Desktop Application
│   │   └── (Electron app)
│   │
│   ├── editing/                        # TRACK G: Editing Dinámica
│   │   ├── incremental_solver.py       # Solver incremental
│   │   └── change_propagation.py       # Propagación de cambios
│   │
│   └── educational/                    # TRACK I: Sistema Educativo
│       └── zettelkasten/               # Zettelkasten
│           ├── dominios/               # Dominios
│           ├── conceptos/              # Conceptos
│           ├── categorias/             # Categorías
│           └── isomorfismos/           # Isomorfismos
│
├── track-i-educational-multidisciplinary/  # Track I (mantener separado)
│   ├── zettelkasten/                   # Sistema de conocimiento
│   ├── docs/                           # Documentación Track I
│   └── ...
│
├── tests/                              # Tests
│   ├── unit/                           # Tests unitarios
│   ├── integration/                    # Tests de integración
│   ├── e2e/                            # Tests end-to-end
│   └── benchmarks/                     # Benchmarks
│
├── examples/                           # Ejemplos de uso
│   ├── basic_csp.py                    # CSP básico
│   ├── with_strategies.py              # Con estrategias
│   ├── with_ml_acceleration.py         # Con aceleración ML
│   └── full_pipeline.py                # Pipeline completo
│
├── scripts/                            # Scripts de utilidad
│   ├── train_mini_ias.py               # Entrenar Mini-IAs
│   ├── benchmark_suite.py              # Benchmarks
│   └── generate_docs.py                # Generar documentación
│
├── README.md                           # README principal (actualizado)
├── PROJECT_STRUCTURE.md                # Este archivo
├── ESTADO_ACTUAL_Y_ROADMAP.md          # Estado y roadmap
├── pyproject.toml                      # Configuración del proyecto
├── setup.py                            # Setup
└── LICENSE                             # Licencia MIT
```

---

## Convenciones de Nomenclatura

### Archivos y Módulos
- **snake_case** para archivos Python: `csp_problem.py`, `ml_accelerator.py`
- **PascalCase** para clases: `SolverOrchestrator`, `MLAccelerator`
- **UPPER_CASE** para constantes: `MAX_ITERATIONS`, `DEFAULT_TIMEOUT`

### Estrategias
- Sufijo `Strategy` para todas las estrategias: `TopologicalAnalysisStrategy`
- Prefijo que indica el tipo: `Analysis`, `Heuristic`, `Propagation`, etc.

### Mini-IAs
- Sufijo `MiniIA` o nombre descriptivo: `VariableSelector`, `PersistencePredictor`
- Agrupadas en suites: `arc_engine_suite.py`, `tda_suite.py`

---

## Flujo de Dependencias

```
Aplicación (Web/Desktop/CLI)
    ↓
InferenceEngine (Track D)
    ↓
SolverOrchestrator (Capa 3)
    ↓
Estrategias (Capa 2) ← MLAccelerator (Capa 5)
    ↓
Núcleo (Capa 1)
```

---

## Archivos Archivados

Los siguientes archivos/directorios han sido movidos a `.archive/`:

### Directorios de Tracks
- `track-a-core/` → `.archive/old-tracks/`
- `track-b-locales/` → `.archive/old-tracks/`
- `track-c-families/` → `.archive/old-tracks/`
- `track-d-inference/` → `.archive/old-tracks/`
- `track-e-web/` → `.archive/old-tracks/`
- `track-f-desktop/` → `.archive/old-tracks/`
- `track-g-editing/` → `.archive/old-tracks/`
- `track-h-formal-math/` → `.archive/old-tracks/`
- Archivos `.tar.gz` de tracks

### Documentos Obsoletos
- `COORDINACION_TRACKS_V3.md` → `.archive/old-docs/`
- `Analisis_Dependencias_Tracks.md` → `.archive/old-docs/`
- `TRACK_B_ENTREGABLE_README.md` → `.archive/old-docs/`

**Razón:** Estos archivos eran parte del sistema de desarrollo por tracks separados.
Con la arquitectura v8.0 unificada, ya no son necesarios.

---

## Track I (Educativo) - Excepción

El directorio `track-i-educational-multidisciplinary/` se **mantiene separado** porque:

1. Es un sistema de conocimiento independiente (Zettelkasten)
2. Tiene su propia estructura y flujo de trabajo
3. No se integra directamente en el código de LatticeWeaver
4. Sirve como documentación educativa y ejemplos

---

## Próximos Pasos

1. **Implementar estrategias base** en `strategies/`
2. **Crear SolverOrchestrator** en `core/orchestrator.py`
3. **Cerrar gaps críticos** (Gap 1-4)
4. **Implementar primeras Mini-IAs** en `ml/mini_nets/`
5. **Actualizar tests** para nueva arquitectura

---

**Versión:** 8.0-alpha  
**Última actualización:** 15 de Octubre, 2025


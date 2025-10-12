# Track A: Core Engine (ACE) - Guía de Trabajo Autónomo

**Desarrollador:** Dev A  
**Duración:** 8 semanas  
**Dependencias:** Ninguna (puede iniciar inmediatamente)  
**Bloqueante para:** Tracks D, E, F

---

## 🎯 Objetivo del Track

Completar y optimizar el **Adaptive Consistency Engine (ACE)** con:
1. Resolver Issue 1 (backtracking optimizado)
2. Implementar SearchSpaceTracer (captura de evolución)
3. Implementar ExperimentRunner (minería masiva)
4. Optimizaciones avanzadas (motor simbólico, ejecución especulativa)

---

## 📦 Contenido del Entregable

Este entregable contiene TODO lo necesario para trabajar de forma autónoma:

1. ✅ **Proyecto completo** (`Entregable6_LatticeWeaver_v4.2_Fases1-3_COMPLETO.tar.gz`)
2. ✅ **Meta-Principios de Diseño** (`LatticeWeaver_Meta_Principios_Diseño.md`)
3. ✅ **Análisis de Dependencias** (`Analisis_Dependencias_Tracks.md`)
4. ✅ **Estructura Global** (`Estructura_Global_Libreria.md`)
5. ✅ **Plan detallado Track A** (`Track_A_Core_Engine_Plan.md`)
6. ✅ Este README con ruta de trabajo

---

## 🚀 Inicio Rápido

### 1. Descomprimir proyecto

```bash
tar -xzf Entregable6_LatticeWeaver_v4.2_Fases1-3_COMPLETO.tar.gz
cd lattice_weaver_project
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
pip install python-louvain  # Para clustering
```

### 3. Ejecutar tests para validar estado inicial

```bash
pytest tests/unit/test_graph_structures.py -v  # 31 tests ✅
pytest tests/unit/test_clustering.py -v        # 17 tests ✅
pytest tests/unit/test_adaptive_consistency.py::test_ac3_basic -v  # Tests básicos ✅
```

### 4. Crear branch de trabajo

```bash
git checkout -b track-a-core
```

---

## 📋 Ruta de Trabajo Autónomo

### Semana 1: Resolver Issue 1

**Objetivo:** Optimizar backtracking en ACE para N-Reinas

**Tareas:**
1. Leer `Estrategias_Aceleracion_No_Reflejadas.md` (Sección Issue 1)
2. Implementar Forward Checking en lugar de AC-3 completo
3. Implementar caché de dominios por nivel
4. Crear tests para validar optimización

**Archivos a modificar:**
- `lattice_weaver/arc_weaver/adaptive_consistency.py`
- `tests/unit/test_adaptive_consistency.py`

**Criterio de éxito:**
- N-Reinas n=8 resuelve en <100ms
- Todos los tests pasan
- Speedup >10x vs versión actual

**Checkpoint:** Viernes semana 1
- Ejecutar: `pytest tests/unit/test_adaptive_consistency.py -v`
- Ejecutar: `pytest tests/benchmarks/test_benchmark_suite.py::test_nqueens_benchmark -v`
- Validar speedup con benchmarks

---

### Semana 2: SearchSpaceTracer (Parte 1)

**Objetivo:** Implementar captura de eventos de búsqueda

**Tareas:**
1. Crear módulo `lattice_weaver/arc_weaver/search_space_tracer.py`
2. Implementar clases `SearchEvent`, `SearchSpaceTracer`
3. Integrar con `AdaptiveConsistencyEngine`
4. Crear tests unitarios

**Archivos a crear:**
- `lattice_weaver/arc_weaver/search_space_tracer.py` (≈300 líneas)
- `tests/unit/test_search_space_tracer.py` (≈150 líneas)

**Archivos a modificar:**
- `lattice_weaver/arc_weaver/adaptive_consistency.py` (añadir hooks de tracing)
- `lattice_weaver/arc_weaver/__init__.py` (exportar SearchSpaceTracer)

**Criterio de éxito:**
- 11 tipos de eventos capturados
- Exportación a CSV y JSON funcional
- Overhead <5% cuando habilitado
- 15+ tests pasando

**Checkpoint:** Viernes semana 2
- Ejecutar: `pytest tests/unit/test_search_space_tracer.py -v`
- Validar captura con problema pequeño (N-Reinas n=4)

---

### Semana 3: SearchSpaceTracer (Parte 2) + Visualización

**Objetivo:** Implementar visualización de espacio de búsqueda

**Tareas:**
1. Crear módulo `lattice_weaver/visualization/search_space_viz.py`
2. Implementar `SearchSpaceVisualizer`
3. Generar visualizaciones con Plotly
4. Crear tests de integración

**Archivos a crear:**
- `lattice_weaver/visualization/__init__.py`
- `lattice_weaver/visualization/search_space_viz.py` (≈400 líneas)
- `tests/integration/test_search_space_tracing.py` (≈100 líneas)

**Criterio de éxito:**
- 4 tipos de visualizaciones generadas
- Reportes HTML interactivos
- 10+ tests de integración pasando

**Checkpoint:** Viernes semana 3
- Ejecutar: `pytest tests/integration/test_search_space_tracing.py -v`
- Generar reporte HTML de ejemplo y validar visualmente

---

### Semana 4: ExperimentRunner (Parte 1)

**Objetivo:** Implementar framework de experimentos masivos

**Tareas:**
1. Crear módulo `lattice_weaver/arc_weaver/experiment_runner.py`
2. Implementar clases `Experiment`, `ExperimentRunner`
3. Implementar grid search de parámetros
4. Crear tests unitarios

**Archivos a crear:**
- `lattice_weaver/arc_weaver/experiment_runner.py` (≈400 líneas)
- `tests/unit/test_experiment_runner.py` (≈200 líneas)

**Criterio de éxito:**
- Grid search funcional
- Ejecución paralela con ProcessPoolExecutor
- Timeout por experimento
- Exportación a CSV
- 20+ tests pasando

**Checkpoint:** Viernes semana 4
- Ejecutar: `pytest tests/unit/test_experiment_runner.py -v`
- Ejecutar experimento pequeño (3x3 grid) y validar CSV

---

### Semana 5: ExperimentRunner (Parte 2) + Análisis

**Objetivo:** Implementar análisis estadístico de experimentos

**Tareas:**
1. Implementar análisis estadístico en `ExperimentRunner`
2. Crear visualizaciones de resultados
3. Implementar comparación de algoritmos
4. Crear tests de integración

**Archivos a modificar:**
- `lattice_weaver/arc_weaver/experiment_runner.py` (añadir análisis)
- `lattice_weaver/visualization/search_space_viz.py` (añadir gráficos)

**Archivos a crear:**
- `tests/integration/test_experiment_analysis.py` (≈150 líneas)

**Criterio de éxito:**
- Análisis estadístico completo (media, std, percentiles)
- Visualizaciones de heatmaps y curvas
- Comparación multi-algoritmo
- 15+ tests de integración pasando

**Checkpoint:** Viernes semana 5
- Ejecutar: `pytest tests/integration/test_experiment_analysis.py -v`
- Ejecutar experimento completo (10x10 grid) y validar reportes

---

### Semana 6: Motor Simbólico

**Objetivo:** Implementar motor simbólico para optimización

**Tareas:**
1. Crear módulo `lattice_weaver/arc_weaver/optimizations/symbolic_engine.py`
2. Implementar detección de simetrías
3. Implementar representación simbólica de restricciones
4. Crear tests

**Archivos a crear:**
- `lattice_weaver/arc_weaver/optimizations/__init__.py`
- `lattice_weaver/arc_weaver/optimizations/symbolic_engine.py` (≈500 líneas)
- `tests/unit/test_symbolic_engine.py` (≈200 líneas)

**Criterio de éxito:**
- Detección de simetrías funcional
- Speedup 2-5x en problemas simétricos
- 20+ tests pasando

**Checkpoint:** Viernes semana 6
- Ejecutar: `pytest tests/unit/test_symbolic_engine.py -v`
- Benchmark con N-Reinas (altamente simétrico)

---

### Semana 7: Ejecución Especulativa

**Objetivo:** Implementar ejecución especulativa

**Tareas:**
1. Crear módulo `lattice_weaver/arc_weaver/optimizations/speculative_execution.py`
2. Implementar predicción de ramas prometedoras
3. Implementar rollback eficiente
4. Crear tests

**Archivos a crear:**
- `lattice_weaver/arc_weaver/optimizations/speculative_execution.py` (≈400 líneas)
- `tests/unit/test_speculative_execution.py` (≈150 líneas)

**Criterio de éxito:**
- Predicción de ramas funcional
- Speedup 2-4x en problemas con heurísticas buenas
- 15+ tests pasando

**Checkpoint:** Viernes semana 7
- Ejecutar: `pytest tests/unit/test_speculative_execution.py -v`
- Benchmark con Sudoku (heurísticas efectivas)

---

### Semana 8: Integración y Validación Final

**Objetivo:** Integrar todas las optimizaciones y validar

**Tareas:**
1. Integrar todas las optimizaciones en ACE
2. Ejecutar suite completa de benchmarks
3. Generar documentación final
4. Preparar entrega para Tracks D, E

**Archivos a modificar:**
- `lattice_weaver/arc_weaver/adaptive_consistency.py` (integrar optimizaciones)
- `README.md` (actualizar con nuevas funcionalidades)

**Archivos a crear:**
- `docs/ACE_OPTIMIZATION_GUIDE.md` (≈500 líneas)
- `docs/SEARCH_SPACE_TRACING.md` (≈300 líneas)
- `docs/EXPERIMENT_RUNNER.md` (≈400 líneas)

**Criterio de éxito:**
- Todos los tests pasan (150+ tests)
- Speedup acumulado >10x vs v4.2 inicial
- Documentación completa
- API estable para Tracks D, E

**Checkpoint:** Viernes semana 8 (SYNC POINT 1)
- Ejecutar: `pytest tests/ -v` (todos los tests)
- Ejecutar: `pytest tests/benchmarks/ -v --benchmark`
- Generar reporte de speedups
- **Reunión de sincronización con todos los devs**
- Entregar ACE completo a Dev A (para Track D)

---

## 🔗 Puntos de Sincronización

### Sync Point 1: Semana 8 (CRÍTICO)

**Fecha:** Viernes semana 8, 15:00

**Participantes:** Dev A, Dev B, Dev C, Tech Lead

**Entregables de Track A:**
1. ✅ ACE completo y optimizado
2. ✅ SearchSpaceTracer funcional
3. ✅ ExperimentRunner funcional
4. ✅ Motor simbólico implementado
5. ✅ Ejecución especulativa implementada
6. ✅ Documentación completa
7. ✅ API estable

**Agenda:**
1. Demo de ACE optimizado (15min)
2. Revisión de API para Tracks D, E (20min)
3. Discusión de issues encontrados (15min)
4. Planificación Track D (10min)

**Acciones post-sync:**
- Dev A: Iniciar Track D (Inference Engine)
- Dev B: Continuar Track B (Locales/Frames)
- Dev C: Continuar Track H (Formal Math)

---

## 📚 Documentos de Referencia

### Lectura Obligatoria (Antes de empezar)

1. **Meta-Principios de Diseño** (`LatticeWeaver_Meta_Principios_Diseño.md`)
   - Leer completo (≈1h)
   - Enfocarse en: Economía Computacional, Dinamismo Adaptativo, No Redundancia

2. **Análisis de Dependencias** (`Analisis_Dependencias_Tracks.md`)
   - Leer secciones: Track A, Sync Point 1, Interfaces A↔D, A↔E
   - Tiempo: 30min

3. **Estructura Global** (`Estructura_Global_Libreria.md`)
   - Leer secciones: Nivel 1-2, Track A (futuro), Grafo de Dependencias
   - Tiempo: 30min

4. **Plan Track A** (`Track_A_Core_Engine_Plan.md`)
   - Leer completo (≈2h)
   - Es tu guía principal

5. **Estrategias de Aceleración** (en `Entregable8_FINAL_Planificacion_Completa.tar.gz`)
   - Leer secciones: Issue 1, Motor Simbólico, Ejecución Especulativa
   - Tiempo: 1h

**Tiempo total de lectura:** ≈5h (hacer en Semana 0 o Día 1)

### Consulta Durante Desarrollo

- **Código actual:** Revisar `lattice_weaver/arc_weaver/` para entender implementación actual
- **Tests actuales:** Revisar `tests/unit/test_adaptive_consistency.py` para entender casos de uso
- **Benchmarks:** Revisar `tests/benchmarks/` para entender cómo medir rendimiento

---

## 🎯 Criterios de Éxito del Track

Al finalizar la Semana 8, debes tener:

✅ **Issue 1 resuelto:** N-Reinas n=8 en <100ms  
✅ **SearchSpaceTracer:** Captura completa de evolución  
✅ **ExperimentRunner:** Minería masiva funcional  
✅ **Motor Simbólico:** Speedup 2-5x en problemas simétricos  
✅ **Ejecución Especulativa:** Speedup 2-4x con heurísticas  
✅ **Speedup acumulado:** >10x vs v4.2 inicial  
✅ **Tests:** 150+ tests pasando  
✅ **Documentación:** Completa y actualizada  
✅ **API estable:** Lista para Tracks D, E

---

## 🚨 Gestión de Bloqueos

### Si encuentras un bloqueo técnico:

1. **Documentar el problema** (30min máximo intentando resolver)
2. **Buscar en documentación** (Meta-Principios, Plan Track A)
3. **Preguntar en Slack** (`#track-a-core`)
4. **Si no se resuelve en 2h:** Escalar a Tech Lead

### Si necesitas cambiar la API:

1. **Evaluar impacto** en Tracks D, E (revisar `Analisis_Dependencias_Tracks.md`)
2. **Proponer cambio** en `#sync-general`
3. **Esperar aprobación** de Tech Lead y Devs afectados
4. **Documentar cambio** en `CHANGELOG.md`

### Si te adelantas al cronograma:

1. **Validar calidad** (tests, documentación)
2. **Ayudar a otros tracks** (code reviews)
3. **Empezar Track D** (si Semana 8 completa)

### Si te atrasas:

1. **Comunicar inmediatamente** en reunión semanal
2. **Identificar causa** (técnica, planificación, externa)
3. **Ajustar plan** con Tech Lead
4. **Priorizar tareas críticas** (Issue 1, API estable)

---

## 📞 Comunicación

### Canales

- **Slack `#track-a-core`:** Preguntas técnicas, progreso diario
- **Slack `#sync-general`:** Coordinación con otros tracks
- **GitHub:** Code reviews, issues, PRs
- **Reunión semanal:** Viernes 14:00-15:00

### Reportes de Progreso

**Formato diario (Slack):**
```
✅ Completado hoy: [tarea]
🚧 En progreso: [tarea]
⏭️ Mañana: [tarea]
🚨 Bloqueos: [ninguno/descripción]
```

**Formato semanal (Reunión):**
- Progreso vs plan (%)
- Tests pasando
- Bloqueos
- Próxima semana

---

## 🏆 Conclusión

Este entregable contiene **TODO** lo necesario para trabajar de forma autónoma en el Track A durante 8 semanas.

**Éxito del track = Éxito del proyecto**

Track A es **crítico** porque bloquea Tracks D, E, F. Tu trabajo permite que otros devs continúen.

**¡Buena suerte!**

---

**Autor:** Equipo LatticeWeaver  
**Fecha:** Diciembre 2024  
**Versión:** 1.0


---

## 🤖 Protocolo de Ejecución Autónoma

Este track sigue el **Protocolo de Ejecución Autónoma e Iterativa**. Ver `PROTOCOLO_EJECUCION_AUTONOMA.md` para detalles completos.

### Resumen del Ciclo

1. **Leer especificación** de la semana
2. **Validar contra principios** de diseño
3. **Implementar funcionalidad**
4. **Implementar tests** (cobertura >90%)
5. **Ejecutar tests**
6. **Analizar fallos** (si >5% fallan)
7. **Generar entregable** incremental
8. **Presentar resultados**
9. **Checkpoint de validación** (timeout 5min)
10. **Continuar o pausar** según criterios

### Condiciones de Parada

- **Sync Point:** Semana 8 (CRÍTICO)
- **Fallos críticos:** >50% tests fallando O >3 fallos críticos
- **Fin de track:** Semana 8 completada

### Continuación Automática

Si después de 5 minutos no hay respuesta y NO es sync point, continuar automáticamente con siguiente semana.

### Formato de Entregable

Cada semana genera un entregable con:
- Código nuevo/modificado
- Tests
- Resultados de tests
- Análisis de fallos (si hay)
- Documentación
- Resumen ejecutivo

Ver `PROTOCOLO_EJECUCION_AUTONOMA.md` para template completo.

---

## 📝 Presentación de Resultados

### Formato Requerido

```markdown
# Entregable Track A - Semana X: [Nombre de la tarea]

**Desarrollador:** Dev A
**Fecha:** [YYYY-MM-DD]
**Duración:** [X] horas
**Estado:** [✅ Completado / ⚠️ Con issues / ❌ Bloqueado]

## Resumen Ejecutivo
[Descripción breve]

## Objetivos Cumplidos
[Tabla de objetivos vs estado]

## Métricas
[Código, tests, tiempo]

## Issues Encontrados
[Lista de issues con severidad]

## Archivos Entregados
[Estructura del entregable]

## Próximos Pasos
[Qué sigue]

## Validación Requerida
[Preguntas para el usuario]
```

### Métricas Obligatorias

- Archivos creados/modificados
- Líneas de código nuevas
- Tests totales/pasando/fallando
- Cobertura de código (%)
- Tiempo de implementación

### Análisis de Fallos

Si >5% de tests fallan, incluir `analisis_fallos.md` con:
- Clasificación de fallos
- Análisis detallado de cada fallo
- Causas raíz
- Soluciones propuestas
- Decisión de continuación

---

## ✅ Checklist Semanal

Antes de presentar entregable:

- [ ] Código implementado según especificación
- [ ] Principios de diseño validados
- [ ] Tests implementados (cobertura >90%)
- [ ] Tests ejecutados
- [ ] Fallos analizados (si >5%)
- [ ] Entregable generado
- [ ] Resumen ejecutivo escrito
- [ ] Métricas capturadas
- [ ] Documentación completa

---

**¡Listo para empezar!** 🚀

Ejecutar Semana 1 y seguir protocolo de ejecución autónoma.

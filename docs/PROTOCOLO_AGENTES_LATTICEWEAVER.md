# Protocolo de Desarrollo para Agentes de LatticeWeaver

**Versión:** 4.0  
**Fecha:** 16 de Octubre, 2025  
**Propósito:** Establecer un protocolo de desarrollo riguroso para agentes autónomos que garantice la creación de código fiable, robusto y eficiente desde el principio, alineado con los meta-principios de diseño de LatticeWeaver.

---

## 🎯 Objetivo

Este protocolo tiene como objetivo guiar a los agentes en un ciclo de desarrollo que prioriza la planificación y el diseño en profundidad, la implementación de código de alta calidad siguiendo patrones de diseño establecidos, el análisis riguroso de errores, la optimización continua y la actualización segura del repositorio mediante merges controlados, asegurando la coherencia y la integridad del repositorio principal.

---

## 📋 Ciclo de Desarrollo para Agentes

### Fase 0: Verificación del Estado del Proyecto

**OBLIGATORIO antes de iniciar cualquier tarea:**

1. **Lectura de Documentación Central**: El agente DEBE leer y comprender:
   - `docs/PROJECT_OVERVIEW.md`: Estado actual del proyecto, arquitectura y prioridades
   - `README.md`: Visión general y guía de inicio
   - Documentación específica de la tarea asignada

2. **Verificación de Coherencia**: El agente DEBE verificar que:
   - La documentación refleja el estado actual del código
   - No existen contradicciones entre documentos
   - Los avances previos están correctamente registrados

3. **Identificación de Contexto**: El agente DEBE identificar:
   - Tareas relacionadas completadas previamente
   - Documentos de diseño/análisis existentes para la tarea
   - Dependencias con otros módulos

**Si se detectan inconsistencias, el agente DEBE reportarlas antes de proceder.**

---

### Fase 1: Planificación y Diseño en Profundidad

1. **Planificación Detallada de la Tarea**: Antes de escribir cualquier código, el agente debe realizar una planificación en profundidad de la tarea, descomponiéndola en subtareas manejables y estimando el esfuerzo requerido.

2. **Diseño Acorde a Principios y Patrones**: El agente debe diseñar una solución que:
   - Se alinee con los **Meta-Principios de Diseño de LatticeWeaver** (`MASTER_DESIGN_PRINCIPLES.md`)
   - Aplique **patrones de diseño establecidos** para garantizar modularidad:
     - **Strategy Pattern**: Para algoritmos intercambiables (solvers, heurísticas)
     - **Factory Pattern**: Para creación de objetos complejos
     - **Observer Pattern**: Para sistemas reactivos
     - **Adapter Pattern**: Para integración de componentes heterogéneos
     - **Facade Pattern**: Para simplificar interfaces complejas
     - **Dependency Injection**: Para desacoplamiento y testabilidad
   - Justifique las decisiones de diseño y cómo se respetan los principios de eficiencia, modularidad, no redundancia, etc.

3. **Documento de Diseño Centralizado**:
   - Si existe un documento de diseño previo para la tarea, el agente DEBE **actualizarlo e integrarlo**, NO crear uno nuevo
   - El documento DEBE incluir:
     - Historial de cambios con fechas y versiones
     - Sección de "Decisiones de Diseño" que explique por qué se eligieron ciertos patrones
     - Diagrama de arquitectura (si aplica)
     - Análisis de trade-offs
   - Formato: `docs/<modulo>/<tarea>_design_vX.Y.md` donde X.Y se incrementa con cada actualización

---

### Fase 2: Implementación y Pruebas

1. **Implementación de Código Modular**: El agente debe implementar el código:
   - Aplicando los patrones de diseño identificados en Fase 1
   - Asegurando alta cohesión y bajo acoplamiento
   - Con interfaces claras y bien definidas
   - Legible, bien documentado y robusto

2. **Pruebas Rigurosas**: Se deben desarrollar tests:
   - Unitarios con alta cobertura (>90%)
   - De integración para validar interacciones entre módulos
   - De patrones (verificar que los patrones de diseño funcionan correctamente)
   - Con fixtures reutilizables en `conftest.py`

3. **Revisión de Librerías y Compatibilidad**: Antes de escribir nuevo código, es mandatorio:
   - Revisar la documentación y el código de las librerías y módulos existentes
   - Verificar compatibilidad y uso correcto de APIs
   - Prevenir duplicación de funcionalidades
   - Respetar el nombre exacto de las funciones

---

### Fase 3: Análisis de Errores y Refinamiento

1. **Análisis en Profundidad de Errores**: Si surgen errores durante las pruebas:
   - Después de 2-3 intentos de corrección, el agente DEBE detenerse
   - Realizar un análisis en profundidad de la causa raíz del problema
   - Evitar cambiar el código arbitrariamente solo para que los tests pasen
   - Documentar el análisis en un archivo `docs/analisis_error_<descripcion>.md`

2. **Refinamiento del Código**: Una vez que el código es funcional:
   - Realizar análisis de eficiencia de los algoritmos
   - Proponer mejoras alineadas con los meta-principios
   - Evaluar integración con el resto de la estructura existente
   - Verificar que los patrones de diseño se aplican correctamente

---

### Política de Resolución de Errores en Testing

Cuando se encuentren errores durante la fase de pruebas, se debe seguir la siguiente política para evitar la introducción de errores catastróficos y mantener la integridad del código base:

1. **Priorizar el ajuste de los casos de prueba**: Antes de realizar cualquier cambio en el código fuente, se debe analizar si el error reside en el caso de prueba mismo. Es común que los tests no reflejen correctamente la lógica esperada o no estén actualizados. En estos casos, se debe corregir el test para que se alinee con el comportamiento correcto del código existente.

2. **Modificar el código solo cuando sea necesario**: Solo se debe proceder a modificar el código fuente cuando se haya verificado que los casos de prueba son correctos y que el error revela un problema real en la lógica de la implementación. Esta medida previene cambios innecesarios que puedan desestabilizar otras partes del sistema.

3. **Análisis de impacto**: Antes de aplicar cualquier corrección en el código, se debe realizar un análisis de impacto para entender cómo el cambio puede afectar a otras librerías o módulos. El objetivo es evitar introducir regresiones o efectos secundarios no deseados.

---

### Fase 4: Documentación y Actualización del Repositorio

#### 4.1 Documentación Exhaustiva

El código funcional debe ser debidamente documentado:

1. **Código**:
   - Docstrings completos en todos los módulos, clases y funciones
   - Comentarios explicativos en lógica compleja
   - Type hints en todas las firmas de funciones

2. **Documentos de Tarea Centralizados**:
   - **REGLA CRÍTICA**: NO crear múltiples documentos para la misma tarea
   - Si existe un documento previo, ACTUALIZARLO integrando la nueva información
   - Estructura de documento de tarea:
     ```markdown
     # [Nombre de la Tarea]
     
     **Versión:** X.Y
     **Última Actualización:** [Fecha]
     **Estado:** [En Progreso / Completado / Bloqueado]
     
     ## Historial de Cambios
     - vX.Y ([Fecha]): [Descripción de cambios]
     - vX.Y-1 ([Fecha]): [Descripción de cambios previos]
     
     ## Objetivo
     [Descripción actualizada del objetivo]
     
     ## Estado Actual
     [Resumen del progreso actual]
     
     ## Diseño
     [Decisiones de diseño actualizadas]
     
     ## Implementación
     [Detalles de implementación]
     
     ## Tests y Cobertura
     [Resultados de tests]
     
     ## Próximos Pasos
     [Tareas pendientes]
     ```

3. **Actualización de Documentación Central** (OBLIGATORIO):
   - **`docs/PROJECT_OVERVIEW.md`**: Actualizar con nuevos módulos, cambios arquitecturales, estado del proyecto
   - **`README.md`**: Actualizar si hay cambios en instalación, uso o características principales
   - Verificar que ambos documentos reflejan el estado actual del repositorio

#### 4.2 Protocolo de Actualización Segura del Repositorio

**POLÍTICA DE MERGE SEGURO (OBLIGATORIO):**

1. **Trabajar en Rama de Integración**:
   ```bash
   git checkout -b integration/<nombre-tarea>
   ```

2. **Antes de Merge a Main**:
   - Ejecutar suite completa de tests: `pytest tests/ -v`
   - Verificar cobertura: `pytest --cov=lattice_weaver --cov-report=term`
   - Verificar que no hay conflictos: `git fetch origin main && git merge origin/main --no-commit --no-ff`

3. **Resolución de Conflictos**:
   - **Análisis de conflictos**: Crear documento `docs/analisis_conflictos_<tarea>.md`
   - **Estrategia de resolución**: Documentar decisiones tomadas
   - **Merge selectivo**: Preservar estructura completa, evitar pérdida de funcionalidad
   - **Validación post-merge**: Ejecutar tests nuevamente

4. **Commit y Push**:
   - Commits descriptivos siguiendo convención:
     ```
     <tipo>(<scope>): <descripción corta>
     
     <descripción detallada>
     
     - Cambio 1
     - Cambio 2
     
     <información adicional>
     ```
   - Tipos: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
   - Merge a main solo cuando tests pasen al 100%

5. **Actualización de Documentación Post-Merge**:
   - Actualizar `PROJECT_OVERVIEW.md` con cambios integrados
   - Actualizar `README.md` si aplica
   - Crear entrada en `CHANGELOG.md` (si existe)

---

## 🚀 Flujo de Trabajo Detallado

```
┌─────────────────────────────────────────────────────────────┐
│ INICIO DE TAREA                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 0. VERIFICACIÓN DEL ESTADO DEL PROYECTO (OBLIGATORIO)      │
│    - Leer PROJECT_OVERVIEW.md y README.md                  │
│    - Verificar coherencia de documentación                 │
│    - Identificar tareas relacionadas y contexto            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. PLANIFICACIÓN Y DISEÑO EN PROFUNDIDAD                    │
│    - Descomponer tarea en subtareas                         │
│    - Diseñar solución con patrones de diseño establecidos   │
│    - Actualizar/crear documento de diseño centralizado      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. IMPLEMENTACIÓN Y PRUEBAS                                 │
│    - Revisar librerías existentes                           │
│    - Implementar código modular con patrones                │
│    - Desarrollar tests con alta cobertura (>90%)            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. EJECUTAR TESTS                                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────┴───────┐
                    │ ¿Tests pasan? │
                    └───────┬───────┘
                            │
                ┌───────────┴───────────┐
                │                       │
               SÍ                      NO
                │                       │
                ▼                       ▼
    ┌─────────────────────┐  ┌──────────────────────────┐
    │ 4A. ANÁLISIS DE       │  │ 4B. ANÁLISIS DE ERRORES    │
    │ ALGORITMOS Y MEJORAS  │  │ (SEGÚN POLÍTICA)         │
    └─────────────────────┘  └──────────────────────────┘
                │                       │
                ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. DOCUMENTACIÓN CENTRALIZADA                               │
│    - Actualizar (NO crear nuevo) documento de tarea         │
│    - Documentar código con docstrings y comentarios         │
│    - OBLIGATORIO: Actualizar PROJECT_OVERVIEW.md            │
│    - OBLIGATORIO: Actualizar README.md si aplica            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. ACTUALIZACIÓN SEGURA DEL REPOSITORIO                    │
│    - Crear rama de integración                              │
│    - Verificar conflictos con main                          │
│    - Merge selectivo preservando funcionalidad              │
│    - Validar tests post-merge (100% pasando)                │
│    - Actualizar documentación central post-merge            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. VERIFICACIÓN FINAL                                       │
│    - Comprobar PROJECT_OVERVIEW.md refleja cambios          │
│    - Comprobar README.md actualizado si aplica              │
│    - Validar que avance está centralizado y operativo       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ FIN DE TAREA                                                │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ Checklist de Finalización de Tarea

Antes de dar por finalizada una tarea, el agente debe verificar que se han cumplido los siguientes puntos:

### Diseño y Planificación
- [ ] Se ha leído y comprendido `PROJECT_OVERVIEW.md` y `README.md`
- [ ] Se ha verificado la coherencia de la documentación existente
- [ ] Se ha realizado una planificación y diseño en profundidad
- [ ] Se han aplicado patrones de diseño apropiados para garantizar modularidad
- [ ] Se ha actualizado (NO creado nuevo) el documento de diseño de la tarea

### Implementación y Testing
- [ ] Se ha realizado una revisión de las librerías existentes para asegurar la compatibilidad
- [ ] El código implementado es funcional, modular, robusto y está bien documentado
- [ ] Se han desarrollado tests con alta cobertura (>90%) y todos pasan
- [ ] Se ha seguido la política de resolución de errores en testing

### Optimización
- [ ] Se ha realizado un análisis de eficiencia de los algoritmos y se han propuesto mejoras
- [ ] Se ha evaluado la integración con el resto de la estructura existente
- [ ] Se ha verificado que los patrones de diseño se aplican correctamente

### Documentación (CRÍTICO)
- [ ] Se ha actualizado el documento de tarea existente (NO se creó uno nuevo)
- [ ] Se ha actualizado `docs/PROJECT_OVERVIEW.md` con los cambios realizados
- [ ] Se ha actualizado `README.md` si hay cambios en instalación, uso o características
- [ ] Se ha verificado que la documentación refleja el estado actual del repositorio
- [ ] Se ha verificado que el avance del proyecto queda completamente reflejado y centralizado

### Actualización del Repositorio
- [ ] Se ha trabajado en rama de integración
- [ ] Se ha verificado ausencia de conflictos con `main`
- [ ] Se ha realizado merge seguro preservando funcionalidad completa
- [ ] Se han ejecutado todos los tests post-merge (100% pasando)
- [ ] Se ha actualizado toda la documentación relevante sin crear conflictos
- [ ] Se ha creado commit descriptivo siguiendo convenciones

---

## 📐 Patrones de Diseño Obligatorios

Los siguientes patrones de diseño DEBEN ser considerados y aplicados cuando sean apropiados:

### Patrones Creacionales
- **Factory Method**: Para crear instancias de solvers, heurísticas, etc.
- **Abstract Factory**: Para familias de objetos relacionados
- **Builder**: Para construcción de objetos complejos paso a paso
- **Singleton**: Solo cuando sea estrictamente necesario (evitar abuso)

### Patrones Estructurales
- **Adapter**: Para integrar componentes con interfaces incompatibles
- **Facade**: Para simplificar interfaces complejas de subsistemas
- **Composite**: Para estructuras jerárquicas (árboles de restricciones, etc.)
- **Decorator**: Para añadir funcionalidad dinámicamente

### Patrones de Comportamiento
- **Strategy**: Para algoritmos intercambiables (OBLIGATORIO en solvers)
- **Observer**: Para sistemas reactivos y notificaciones
- **Template Method**: Para definir esqueletos de algoritmos
- **Command**: Para encapsular operaciones
- **State**: Para objetos con comportamiento dependiente del estado

### Patrones Específicos de LatticeWeaver
- **Dependency Injection**: OBLIGATORIO para desacoplamiento y testabilidad
- **Repository Pattern**: Para acceso a datos y caché
- **Unit of Work**: Para transacciones y consistencia

**Justificación Requerida**: Si un patrón estándar NO se aplica, el agente DEBE justificar por qué en el documento de diseño.

---

## 📊 Gestión de Documentación Centralizada

### Principio de Documento Único por Tarea

**REGLA DE ORO**: Una tarea = Un documento que evoluciona

- ❌ **INCORRECTO**: `analisis_v1.md`, `analisis_v2.md`, `analisis_final.md`
- ✅ **CORRECTO**: `analisis.md` con historial de versiones interno

### Estructura de Versionado Interno

```markdown
# [Nombre del Documento]

**Versión Actual:** 3.2  
**Última Actualización:** 16 de Octubre, 2025

## Historial de Cambios

### v3.2 (16 Oct 2025)
- Añadido análisis de rendimiento con nuevos benchmarks
- Corrección de errores en sección de implementación
- Integración de feedback de tests

### v3.1 (15 Oct 2025)
- Actualización de diseño arquitectural
- Añadido patrón Strategy para solvers

### v3.0 (14 Oct 2025)
- Refactorización completa del documento
- Integración de versiones anteriores
- [Contenido de v2.x integrado y corregido]

[Contenido del documento...]
```

### Ubicación de Documentos

```
docs/
├── PROJECT_OVERVIEW.md          # Documento central del proyecto (ACTUALIZAR SIEMPRE)
├── PROTOCOLO_AGENTES_LATTICEWEAVER.md
├── MASTER_DESIGN_PRINCIPLES.md
├── <modulo>/
│   ├── <tarea>_design.md        # Documento de diseño (versión única)
│   ├── <tarea>_analysis.md      # Documento de análisis (versión única)
│   └── <tarea>_roadmap.md       # Roadmap de la tarea (versión única)
└── analisis_conflictos/         # Solo para merges complejos
    └── merge_<fecha>_<descripcion>.md
```

---

## 🔄 Protocolo de Actualización de PROJECT_OVERVIEW.md

**OBLIGATORIO al finalizar cualquier tarea que:**
- Añada nuevos módulos o componentes
- Modifique arquitectura existente
- Complete hitos del roadmap
- Cambie prioridades o estado del proyecto

### Secciones a Actualizar

1. **Fecha de Actualización**: Cambiar a fecha actual
2. **Versión del Repositorio**: Incrementar si aplica
3. **Resumen Ejecutivo**: Actualizar si hay cambios significativos
4. **Componentes Clave**: Añadir/actualizar módulos nuevos o modificados
5. **Hoja de Ruta Estratégica**: Marcar hitos completados, actualizar prioridades
6. **Estado de Tracks**: Actualizar progreso de tracks específicos

### Formato de Actualización

```markdown
## [Sección Relevante]

**Última actualización:** [Fecha] - [Breve descripción del cambio]

[Contenido actualizado...]

---
**Cambios recientes:**
- ([Fecha]) [Descripción del cambio 1]
- ([Fecha]) [Descripción del cambio 2]
```

---

## 🚨 Violaciones del Protocolo

Las siguientes acciones se consideran **violaciones graves** del protocolo:

1. ❌ No leer `PROJECT_OVERVIEW.md` antes de iniciar tarea
2. ❌ Crear múltiples documentos para la misma tarea sin integrar versiones anteriores
3. ❌ No actualizar `PROJECT_OVERVIEW.md` después de cambios significativos
4. ❌ Hacer merge a `main` sin ejecutar tests completos
5. ❌ No aplicar patrones de diseño sin justificación
6. ❌ Ignorar conflictos de merge o resolverlos arbitrariamente
7. ❌ No documentar decisiones de diseño importantes

**Consecuencia**: La tarea debe ser revertida y rehecha siguiendo el protocolo.

---

**Este protocolo es de obligado cumplimiento para todos los agentes que contribuyan al desarrollo de LatticeWeaver.**

**Versión 4.0 - Cambios principales:**
- Añadida Fase 0: Verificación del Estado del Proyecto
- Protocolo de Merge Seguro obligatorio
- Gestión de documentación centralizada (documento único por tarea)
- Actualización obligatoria de PROJECT_OVERVIEW.md y README.md
- Patrones de diseño obligatorios para modularidad
- Verificación post-lectura del estado del proyecto


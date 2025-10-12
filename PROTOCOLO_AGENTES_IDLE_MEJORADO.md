# Protocolo de Agentes Idle - Versión Mejorada

**Versión:** 2.0  
**Fecha:** 12 de Octubre, 2025  
**Propósito:** Gestión inteligente de agentes en estado idle con asignación proactiva de tareas

---

## 📋 Estados de Agentes

### Estados Posibles

1. **ACTIVE** - Trabajando en tarea asignada de su track
2. **WAITING** - Esperando dependencia (timeout < 30 segundos)
3. **IDLE** - Sin tareas asignadas o bloqueado por dependencias
4. **SYNC_WAITING** - Esperando sync point
5. **BLOCKED** - Bloqueado por issues críticos

---

## 🎯 Sistema de Prioridades para Agentes Idle

Cuando un agente entra en estado IDLE, el sistema asigna tareas según esta jerarquía de prioridades:

### Nivel 1: Apoyo a Track I (Visualizador Educativo) - PRIORIDAD MÁXIMA

**Objetivo:** Acelerar el desarrollo del visualizador educativo que beneficia a todo el proyecto.

**Tareas disponibles:**

```python
TRACK_I_SUPPORT_TASKS = [
    {
        "id": "IDLE-I-DOC-001",
        "title": "Documentación de API de visualización",
        "type": "DOCUMENTATION",
        "priority": "HIGH",
        "estimated_hours": 4,
        "skills_required": ["python", "documentation"],
        "description": "Documentar API completa del módulo de visualización con ejemplos"
    },
    {
        "id": "IDLE-I-TEST-001",
        "title": "Tests unitarios para CSP Visualizer",
        "type": "TESTING",
        "priority": "HIGH",
        "estimated_hours": 6,
        "skills_required": ["python", "pytest", "visualization"],
        "description": "Crear suite completa de tests unitarios para CSP Visualizer"
    },
    {
        "id": "IDLE-I-TEST-002",
        "title": "Tests E2E para Lattice Explorer",
        "type": "TESTING",
        "priority": "HIGH",
        "estimated_hours": 8,
        "skills_required": ["typescript", "playwright", "testing"],
        "description": "Implementar tests end-to-end para Lattice Explorer"
    },
    {
        "id": "IDLE-I-DESIGN-001",
        "title": "Diseño de UI para Topology Viewer",
        "type": "EDUCATIONAL",
        "priority": "MEDIUM",
        "estimated_hours": 6,
        "skills_required": ["ui-design", "visualization"],
        "description": "Diseñar interfaz de usuario para visualizador topológico"
    },
    {
        "id": "IDLE-I-FEATURE-001",
        "title": "Implementar exportación a formatos adicionales",
        "type": "FEATURE",
        "priority": "MEDIUM",
        "estimated_hours": 4,
        "skills_required": ["python", "file-formats"],
        "description": "Añadir soporte para exportar visualizaciones a WebP, AVIF"
    },
    {
        "id": "IDLE-I-TUTORIAL-001",
        "title": "Crear tutorial interactivo de CSP",
        "type": "EDUCATIONAL",
        "priority": "HIGH",
        "estimated_hours": 10,
        "skills_required": ["documentation", "pedagogy", "visualization"],
        "description": "Desarrollar tutorial paso a paso sobre CSP con visualizaciones"
    },
    {
        "id": "IDLE-I-TUTORIAL-002",
        "title": "Crear tutorial de FCA con ejemplos",
        "type": "EDUCATIONAL",
        "priority": "HIGH",
        "estimated_hours": 10,
        "skills_required": ["documentation", "pedagogy", "fca"],
        "description": "Tutorial completo de Formal Concept Analysis con casos de uso"
    },
    {
        "id": "IDLE-I-PERF-001",
        "title": "Optimizar rendering de grafos grandes",
        "type": "OPTIMIZATION",
        "priority": "MEDIUM",
        "estimated_hours": 8,
        "skills_required": ["python", "performance", "algorithms"],
        "description": "Optimizar visualización de grafos con >1000 nodos"
    }
]
```

### Nivel 2: Tareas Encoladas de Otros Tracks

**Objetivo:** Ayudar a otros tracks con tareas pendientes en su backlog.

**Proceso:**

1. Consultar backlogs de todos los tracks activos
2. Filtrar tareas según skills del agente idle
3. Priorizar por:
   - Urgencia (deadline cercano)
   - Impacto (bloquea otros tracks)
   - Complejidad (match con capacidad del agente)

**Ejemplo de consulta:**

```python
def get_enqueued_tasks_from_other_tracks(idle_agent_id: str) -> List[Task]:
    """
    Obtiene tareas encoladas de otros tracks que el agente puede realizar
    """
    idle_agent = load_agent(idle_agent_id)
    all_tracks = ["track-a", "track-b", "track-c", "track-d", 
                  "track-e", "track-f", "track-g", "track-h", "track-i"]
    
    enqueued_tasks = []
    
    for track in all_tracks:
        if track == idle_agent.track:
            continue  # Saltar propio track
        
        track_backlog = load_track_backlog(track)
        
        for task in track_backlog:
            if task.status == "ENQUEUED" and can_agent_do_task(idle_agent, task):
                enqueued_tasks.append(task)
    
    # Ordenar por prioridad y urgencia
    enqueued_tasks.sort(key=lambda t: (t.priority.score(), -t.days_until_deadline))
    
    return enqueued_tasks
```

### Nivel 3: Tareas Proactivas de Mejora del Código

**Objetivo:** Mejorar la calidad, eficiencia y mantenibilidad del código existente.

#### 3.1 Búsqueda de Ineficiencias

**Tareas:**

```python
PROACTIVE_INEFFICIENCY_TASKS = [
    {
        "id": "IDLE-PERF-001",
        "title": "Análisis de performance con profiler",
        "type": "OPTIMIZATION",
        "priority": "MEDIUM",
        "estimated_hours": 4,
        "description": """
        Ejecutar profiler (cProfile, line_profiler) en módulos críticos:
        - arc_engine.core
        - lattice_core.builder
        - topology.tda_engine
        
        Identificar:
        - Funciones con >10% del tiempo total
        - Bucles ineficientes
        - Llamadas repetidas innecesarias
        
        Generar reporte con recomendaciones.
        """
    },
    {
        "id": "IDLE-PERF-002",
        "title": "Análisis de uso de memoria",
        "type": "OPTIMIZATION",
        "priority": "MEDIUM",
        "estimated_hours": 4,
        "description": """
        Ejecutar memory_profiler en módulos críticos.
        
        Identificar:
        - Copias innecesarias de datos
        - Estructuras de datos subóptimas
        - Memory leaks potenciales
        - Oportunidades para object pooling
        
        Generar reporte con recomendaciones.
        """
    },
    {
        "id": "IDLE-PERF-003",
        "title": "Optimización de imports y carga de módulos",
        "type": "OPTIMIZATION",
        "priority": "LOW",
        "estimated_hours": 2,
        "description": """
        Analizar tiempo de carga de módulos:
        - Identificar imports circulares
        - Optimizar imports pesados (lazy loading)
        - Reducir dependencias innecesarias
        
        Medir impacto en tiempo de startup.
        """
    },
    {
        "id": "IDLE-PERF-004",
        "title": "Identificar oportunidades de paralelización",
        "type": "OPTIMIZATION",
        "priority": "MEDIUM",
        "estimated_hours": 6,
        "description": """
        Analizar código secuencial que podría paralelizarse:
        - Bucles independientes
        - Operaciones I/O bound
        - Procesamiento de listas grandes
        
        Estimar speedup potencial.
        Generar propuestas de implementación.
        """
    }
]
```

#### 3.2 Búsqueda de Redundancias

**Tareas:**

```python
PROACTIVE_REDUNDANCY_TASKS = [
    {
        "id": "IDLE-REFACTOR-001",
        "title": "Detectar código duplicado",
        "type": "REFACTORING",
        "priority": "MEDIUM",
        "estimated_hours": 4,
        "description": """
        Usar herramientas de detección de duplicación:
        - pylint --duplicate-code
        - jscpd (para frontend)
        
        Identificar:
        - Bloques de código >10 líneas duplicados
        - Funciones similares en diferentes módulos
        - Patrones repetidos
        
        Proponer refactorizaciones para eliminar duplicación.
        """
    },
    {
        "id": "IDLE-REFACTOR-002",
        "title": "Consolidar funcionalidad redundante",
        "type": "REFACTORING",
        "priority": "MEDIUM",
        "estimated_hours": 6,
        "description": """
        Buscar funcionalidad implementada múltiples veces:
        - Funciones de utilidad duplicadas
        - Validaciones repetidas
        - Conversiones de datos redundantes
        
        Crear módulo de utilidades compartidas.
        Refactorizar para usar versión única.
        """
    },
    {
        "id": "IDLE-REFACTOR-003",
        "title": "Simplificar jerarquías de clases",
        "type": "REFACTORING",
        "priority": "LOW",
        "estimated_hours": 4,
        "description": """
        Analizar jerarquías de herencia:
        - Identificar jerarquías >3 niveles
        - Buscar oportunidades para composición
        - Detectar clases con un solo hijo (innecesarias)
        
        Proponer simplificaciones.
        """
    },
    {
        "id": "IDLE-REFACTOR-004",
        "title": "Eliminar código muerto",
        "type": "REFACTORING",
        "priority": "LOW",
        "estimated_hours": 3,
        "description": """
        Identificar código no utilizado:
        - Funciones nunca llamadas
        - Clases nunca instanciadas
        - Imports no utilizados
        - Variables no leídas
        
        Usar vulture, coverage.py para detectar.
        Crear PR para eliminar código muerto.
        """
    }
]
```

#### 3.3 Búsqueda de Puntos Problemáticos

**Tareas:**

```python
PROACTIVE_PROBLEM_DETECTION_TASKS = [
    {
        "id": "IDLE-QUALITY-001",
        "title": "Análisis de complejidad ciclomática",
        "type": "REFACTORING",
        "priority": "MEDIUM",
        "estimated_hours": 3,
        "description": """
        Ejecutar radon para medir complejidad:
        - Identificar funciones con complejidad >10
        - Identificar módulos con complejidad promedio >5
        
        Proponer refactorizaciones:
        - Extraer funciones
        - Simplificar lógica condicional
        - Aplicar patrones de diseño
        """
    },
    {
        "id": "IDLE-QUALITY-002",
        "title": "Detectar code smells",
        "type": "REFACTORING",
        "priority": "MEDIUM",
        "estimated_hours": 4,
        "description": """
        Buscar anti-patrones comunes:
        - Funciones muy largas (>50 líneas)
        - Clases muy grandes (>500 líneas)
        - Muchos parámetros (>5)
        - Acoplamiento alto
        - Cohesión baja
        
        Generar reporte con recomendaciones.
        """
    },
    {
        "id": "IDLE-QUALITY-003",
        "title": "Análisis de seguridad con bandit",
        "type": "REFACTORING",
        "priority": "HIGH",
        "estimated_hours": 2,
        "description": """
        Ejecutar bandit para detectar vulnerabilidades:
        - Uso inseguro de eval/exec
        - Hardcoded passwords/secrets
        - SQL injection potencial
        - Deserialización insegura
        
        Priorizar y corregir issues críticos.
        """
    },
    {
        "id": "IDLE-QUALITY-004",
        "title": "Revisar manejo de errores",
        "type": "REFACTORING",
        "priority": "MEDIUM",
        "estimated_hours": 4,
        "description": """
        Auditar manejo de excepciones:
        - Bloques try/except vacíos
        - Excepciones genéricas (Exception)
        - Falta de logging en errores
        - Recursos no liberados en errores
        
        Mejorar robustez del código.
        """
    },
    {
        "id": "IDLE-QUALITY-005",
        "title": "Análisis de cobertura de tests",
        "type": "TESTING",
        "priority": "HIGH",
        "estimated_hours": 3,
        "description": """
        Identificar código sin cobertura:
        - Módulos con <70% cobertura
        - Funciones críticas sin tests
        - Casos edge no testeados
        
        Priorizar y crear tests faltantes.
        """
    }
]
```

### Nivel 4: Planificación de Futuras Fases e Hitos

**Objetivo:** Preparar el terreno para desarrollo futuro.

**Tareas:**

```python
PROACTIVE_PLANNING_TASKS = [
    {
        "id": "IDLE-PLAN-001",
        "title": "Diseño de Fase 4: Optimizaciones Avanzadas",
        "type": "FEATURE",
        "priority": "LOW",
        "estimated_hours": 8,
        "description": """
        Diseñar próxima fase de optimizaciones:
        
        Investigar:
        - Compilación JIT con Numba
        - Uso de Cython para módulos críticos
        - GPU acceleration con CuPy
        - Distributed computing con Dask
        
        Deliverables:
        - Documento de diseño
        - Análisis de costo-beneficio
        - Prototipos de concepto
        - Plan de implementación
        """
    },
    {
        "id": "IDLE-PLAN-002",
        "title": "Roadmap de integración con herramientas externas",
        "type": "FEATURE",
        "priority": "LOW",
        "estimated_hours": 6,
        "description": """
        Planificar integraciones futuras:
        
        Herramientas a considerar:
        - Jupyter notebooks
        - VS Code extension
        - CLI avanzada
        - API REST pública
        - Plugins para IDEs
        
        Deliverables:
        - Roadmap de integraciones
        - Priorización
        - Estimaciones de esfuerzo
        """
    },
    {
        "id": "IDLE-PLAN-003",
        "title": "Diseño de sistema de plugins",
        "type": "FEATURE",
        "priority": "MEDIUM",
        "estimated_hours": 10,
        "description": """
        Diseñar arquitectura de plugins:
        
        Objetivos:
        - Permitir extensiones de terceros
        - Sistema de hooks y eventos
        - Gestión de dependencias de plugins
        - Marketplace de plugins
        
        Deliverables:
        - Documento de arquitectura
        - API de plugins
        - Ejemplos de plugins
        - Guía para desarrolladores
        """
    },
    {
        "id": "IDLE-PLAN-004",
        "title": "Planificación de hitos a largo plazo (v6.0-v10.0)",
        "type": "FEATURE",
        "priority": "LOW",
        "estimated_hours": 12,
        "description": """
        Definir visión a largo plazo del proyecto:
        
        Versiones a planificar:
        - v6.0: Optimizaciones y performance
        - v7.0: Integraciones y ecosistema
        - v8.0: Machine learning y AI
        - v9.0: Cloud y distributed computing
        - v10.0: Enterprise features
        
        Deliverables:
        - Roadmap detallado
        - Análisis de mercado
        - Recursos necesarios
        - Timeline estimado
        """
    },
    {
        "id": "IDLE-PLAN-005",
        "title": "Investigación de nuevas técnicas de CSP",
        "type": "FEATURE",
        "priority": "MEDIUM",
        "estimated_hours": 16,
        "description": """
        Investigar estado del arte en CSP:
        
        Áreas a explorar:
        - Nuevos algoritmos de consistencia
        - Técnicas de machine learning para CSP
        - Constraint learning
        - Symmetry breaking avanzado
        - Portfolio solvers
        
        Deliverables:
        - Survey de literatura
        - Análisis de aplicabilidad
        - Prototipos de algoritmos prometedores
        - Propuestas de implementación
        """
    }
]
```

---

## 🔄 Flujo de Asignación de Tareas Idle

```python
def assign_idle_task(agent_id: str) -> Optional[Task]:
    """
    Asigna una tarea a un agente idle según prioridades
    """
    agent = load_agent(agent_id)
    
    # Nivel 1: Apoyo a Track I
    track_i_tasks = get_track_i_support_tasks()
    compatible_tasks = filter_by_skills(track_i_tasks, agent.skills)
    if compatible_tasks:
        return select_best_task(compatible_tasks, agent)
    
    # Nivel 2: Tareas encoladas de otros tracks
    enqueued_tasks = get_enqueued_tasks_from_other_tracks(agent_id)
    if enqueued_tasks:
        return select_best_task(enqueued_tasks, agent)
    
    # Nivel 3: Tareas proactivas de mejora
    proactive_tasks = []
    proactive_tasks.extend(PROACTIVE_INEFFICIENCY_TASKS)
    proactive_tasks.extend(PROACTIVE_REDUNDANCY_TASKS)
    proactive_tasks.extend(PROACTIVE_PROBLEM_DETECTION_TASKS)
    
    compatible_tasks = filter_by_skills(proactive_tasks, agent.skills)
    if compatible_tasks:
        return select_best_task(compatible_tasks, agent)
    
    # Nivel 4: Planificación futura
    planning_tasks = PROACTIVE_PLANNING_TASKS
    compatible_tasks = filter_by_skills(planning_tasks, agent.skills)
    if compatible_tasks:
        return select_best_task(compatible_tasks, agent)
    
    # No hay tareas disponibles
    return None
```

---

## 📊 Tracking de Tareas Idle

### Archivo de Estado: `.agent-status/idle-tasks.json`

```json
{
  "version": "2.0",
  "last_update": "2025-10-12T10:30:00Z",
  "tasks": [
    {
      "id": "IDLE-I-DOC-001",
      "title": "Documentación de API de visualización",
      "type": "DOCUMENTATION",
      "priority": "HIGH",
      "estimated_hours": 4,
      "skills_required": ["python", "documentation"],
      "track_related": "track-i-educational-viz",
      "status": "AVAILABLE",
      "created_at": "2025-10-12T08:00:00Z"
    },
    {
      "id": "IDLE-PERF-001",
      "title": "Análisis de performance con profiler",
      "type": "OPTIMIZATION",
      "priority": "MEDIUM",
      "estimated_hours": 4,
      "skills_required": ["python", "performance"],
      "track_related": "all",
      "status": "IN_PROGRESS",
      "assigned_to": "agent-track-d",
      "assigned_at": "2025-10-12T09:00:00Z"
    }
  ],
  "statistics": {
    "total_tasks": 45,
    "available": 32,
    "in_progress": 8,
    "completed": 5,
    "by_type": {
      "DOCUMENTATION": 8,
      "TESTING": 12,
      "OPTIMIZATION": 10,
      "REFACTORING": 9,
      "FEATURE": 6
    }
  }
}
```

---

## 🎯 Métricas de Éxito

### KPIs para Agentes Idle

1. **Tiempo de idle productivo:** % de tiempo idle gastado en tareas útiles
2. **Tareas completadas:** Número de tareas idle completadas
3. **Impacto:** Mejoras medibles (cobertura, performance, etc.)
4. **Colaboración:** Número de tracks ayudados

### Reporte de Actividad Idle

```bash
python scripts/generate_idle_activity_report.py --agent-id agent-track-d --week 2
```

**Output:**

```
📊 Reporte de Actividad Idle - agent-track-d
Semana: 2
Período: 2025-10-05 a 2025-10-11

⏱️  Tiempo en estado IDLE: 18h (22.5% del tiempo total)

✅ Tareas completadas: 3
   1. IDLE-I-TEST-001: Tests unitarios para CSP Visualizer (6h)
   2. IDLE-PERF-001: Análisis de performance con profiler (4h)
   3. IDLE-REFACTOR-001: Detectar código duplicado (4h)

📈 Impacto:
   - Cobertura de Track I: +12% (de 76% a 88%)
   - Performance de arc_engine: +15% (optimización identificada)
   - Código duplicado reducido: -230 líneas

🤝 Tracks ayudados: 2 (Track I, Track A)

💡 Próximas tareas sugeridas:
   1. IDLE-I-TUTORIAL-001: Tutorial interactivo de CSP
   2. IDLE-QUALITY-003: Análisis de seguridad con bandit
```

---

## ✅ Checklist de Implementación

- [ ] Sistema de prioridades implementado
- [ ] Tareas de apoyo a Track I definidas
- [ ] Sistema de consulta de backlogs implementado
- [ ] Tareas proactivas de mejora definidas
- [ ] Tareas de planificación futura definidas
- [ ] Scripts de asignación automática funcionando
- [ ] Tracking de tareas idle implementado
- [ ] Métricas de éxito definidas
- [ ] Reportes de actividad idle funcionando

---

**¡Sistema de agentes idle optimizado para máxima productividad!** 🚀


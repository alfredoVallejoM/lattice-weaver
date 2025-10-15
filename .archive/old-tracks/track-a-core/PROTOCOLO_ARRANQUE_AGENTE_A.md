# Protocolo de Arranque - Agente Track A (Core Engine)

**Track:** A - Core Engine (ACE)  
**Agente ID:** `agent-track-a`  
**Duración:** 8 semanas  
**Prioridad:** Alta  
**Versión Protocolo:** 1.0

---

## 🚀 Secuencia de Arranque

### Paso 1: Inicialización del Entorno

```bash
# 1.1 Verificar Python y dependencias
python3 --version  # Debe ser >= 3.11
pip3 --version

# 1.2 Crear entorno virtual
cd /workspace/lattice-weaver
python3 -m venv venv-track-a
source venv-track-a/bin/activate

# 1.3 Instalar dependencias base
pip install -r requirements-track-a.txt

# 1.4 Verificar instalación
python -c "import numpy, networkx, pytest; print('✅ Dependencias OK')"
```

### Paso 2: Sincronización con GitHub

```bash
# 2.1 Ejecutar script de sincronización
python scripts/sync_agent.py --agent-id agent-track-a --track track-a-core

# Este script realiza:
# - git fetch origin
# - Verificar estado de la rama track-a-core
# - Comparar con estado local
# - Descargar tar.gz si existe
# - Extraer y verificar integridad
# - Actualizar estado del agente
```

**Script de sincronización:** `scripts/sync_agent.py`

```python
#!/usr/bin/env python3
"""
Script de sincronización del agente con GitHub
"""
import subprocess
import json
import sys
from pathlib import Path

def sync_agent(agent_id: str, track: str):
    print(f"🔄 Sincronizando {agent_id} con track {track}...")
    
    # 1. Fetch desde GitHub
    result = subprocess.run(
        ["git", "fetch", "origin", track],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Error en git fetch: {result.stderr}")
        return False
    
    # 2. Verificar estado local vs remoto
    local_commit = subprocess.run(
        ["git", "rev-parse", track],
        capture_output=True,
        text=True
    ).stdout.strip()
    
    remote_commit = subprocess.run(
        ["git", "rev-parse", f"origin/{track}"],
        capture_output=True,
        text=True
    ).stdout.strip()
    
    if local_commit == remote_commit:
        print("✅ Código local está actualizado")
        return True
    
    # 3. Hay cambios remotos - actualizar
    print(f"📥 Actualizando desde remoto...")
    print(f"   Local:  {local_commit[:8]}")
    print(f"   Remoto: {remote_commit[:8]}")
    
    # 4. Merge o rebase según configuración
    result = subprocess.run(
        ["git", "merge", f"origin/{track}"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Error en merge: {result.stderr}")
        return False
    
    print("✅ Código actualizado correctamente")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--track", required=True)
    args = parser.parse_args()
    
    success = sync_agent(args.agent_id, args.track)
    sys.exit(0 if success else 1)
```

### Paso 3: Verificación de Estado del Proyecto

```bash
# 3.1 Leer estado del track desde GitHub
python scripts/check_track_status.py --track track-a-core

# Este script:
# - Lee .agent-status/agent-track-a.json del repositorio
# - Compara con estado local
# - Muestra progreso actual
# - Identifica siguiente tarea
```

**Formato del archivo de estado:** `.agent-status/agent-track-a.json`

```json
{
  "agent_id": "agent-track-a",
  "track": "track-a-core",
  "version": "1.0",
  "last_update": "2025-10-12T10:30:00Z",
  "progress": {
    "current_week": 1,
    "total_weeks": 8,
    "percentage": 12.5,
    "weeks_completed": 1
  },
  "current_task": {
    "id": "A-W2-T1",
    "title": "Implementar AC-3.1 optimizado",
    "status": "IN_PROGRESS",
    "started_at": "2025-10-12T08:00:00Z"
  },
  "completed_tasks": [
    {
      "id": "A-W1-T1",
      "title": "Setup estructura del proyecto",
      "completed_at": "2025-10-11T18:00:00Z"
    }
  ],
  "flags": {
    "READY_FOR_SYNC": false,
    "WAITING_FOR_DEPENDENCY": false,
    "HAS_CRITICAL_ISSUES": false,
    "IDLE": false
  },
  "metrics": {
    "tests_passed": 45,
    "tests_total": 50,
    "coverage": 87.5,
    "lines_of_code": 2340
  }
}
```

### Paso 4: Cargar Plan de Desarrollo

```bash
# 4.1 Leer plan de desarrollo del track
cat docs/tracks/TRACK_A_PLAN.md

# 4.2 Identificar tarea actual
python scripts/get_current_task.py --agent-id agent-track-a
```

### Paso 5: Iniciar Desarrollo

```bash
# 5.1 Activar modo de desarrollo
python scripts/start_development.py --agent-id agent-track-a

# Este script:
# - Actualiza estado a "ACTIVE"
# - Inicia timer de la tarea
# - Configura entorno de desarrollo
# - Abre editor con archivos relevantes
```

---

## 📋 Estándares de Desarrollo

### Estándar de Código

**Guía de estilo:** PEP 8 + Google Python Style Guide

**Herramientas obligatorias:**
- `black` - Formateo automático
- `isort` - Ordenamiento de imports
- `pylint` - Linting
- `mypy` - Type checking

**Configuración en `pyproject.toml`:**

```toml
[tool.black]
line-length = 100
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 100

[tool.pylint]
max-line-length = 100
disable = ["C0111", "C0103"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

**Pre-commit hooks:**

```bash
# Instalar pre-commit
pip install pre-commit
pre-commit install

# Ejecutar manualmente
pre-commit run --all-files
```

### Estándar de Testing

**Cobertura mínima:** 85%

**Estructura de tests:**

```
tests/
├── unit/
│   └── arc_engine/
│       ├── test_core.py
│       ├── test_domains.py
│       └── test_constraints.py
├── integration/
│   └── test_arc_engine_integration.py
└── benchmarks/
    └── benchmark_arc_engine.py
```

**Nomenclatura de tests:**

```python
def test_<función>_<escenario>_<resultado_esperado>():
    """
    Ejemplo: test_enforce_arc_consistency_with_empty_domain_returns_false
    """
    pass
```

**Ejecutar tests:**

```bash
# Todos los tests
pytest

# Con cobertura
pytest --cov=lattice_weaver.arc_engine --cov-report=html

# Solo tests unitarios
pytest tests/unit/

# Solo tests de integración
pytest tests/integration/
```

### Estándar de Documentación

**Docstrings:** Google Style

```python
def enforce_arc_consistency(self) -> bool:
    """
    Aplica consistencia de arcos en todo el CSP usando AC-3.1 optimizado.
    
    Este método itera sobre todas las restricciones del problema,
    eliminando valores inconsistentes de los dominios de las variables
    hasta alcanzar un punto fijo.
    
    Args:
        None
    
    Returns:
        bool: False si se detecta una inconsistencia (dominio vacío),
              True si se alcanza consistencia de arcos.
    
    Raises:
        ValueError: Si el problema no está inicializado correctamente.
    
    Examples:
        >>> engine = ArcEngine()
        >>> engine.add_variable("x", [1, 2, 3])
        >>> engine.add_variable("y", [1, 2, 3])
        >>> engine.add_constraint("x", "y", lambda a, b: a != b)
        >>> engine.enforce_arc_consistency()
        True
    
    Note:
        Este método modifica los dominios de las variables in-place.
        Para preservar el estado original, crear una copia antes de llamar.
    """
    pass
```

**Anotaciones i18n:**

```python
"""
@i18n:key enforce_arc_consistency
@i18n:desc_es Aplica consistencia de arcos en todo el CSP.
@i18n:desc_en Enforces arc consistency on the entire CSP.
@i18n:desc_fr Applique la cohérence d'arc sur l'ensemble du CSP.
"""
```

### Estándar de Commits

**Formato:** Conventional Commits

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Tipos permitidos:**
- `feat`: Nueva funcionalidad
- `fix`: Corrección de bug
- `docs`: Cambios en documentación
- `style`: Formateo, sin cambios de código
- `refactor`: Refactorización
- `test`: Añadir o modificar tests
- `chore`: Tareas de mantenimiento

**Ejemplos:**

```
feat(arc-engine): implementar AC-3.1 con last support

Añade optimización de last support al algoritmo AC-3.1
para reducir el número de verificaciones de consistencia.

Closes #42
```

```
fix(domains): corregir bug en BitSetDomain.remove()

El método remove() no actualizaba correctamente el contador
de elementos cuando se eliminaba el último bit del conjunto.

Fixes #58
```

---

## 🔄 Ciclo de Desarrollo

### 1. Seleccionar Tarea

```bash
# Obtener siguiente tarea del plan
python scripts/get_next_task.py --agent-id agent-track-a

# Actualizar estado
python scripts/update_status.py --agent-id agent-track-a --status IN_PROGRESS --task-id A-W2-T1
```

### 2. Implementar

```bash
# Crear rama de feature
git checkout -b feature/A-W2-T1-ac31-optimization

# Desarrollar
# ... editar código ...

# Formatear y lint
black lattice_weaver/arc_engine/
isort lattice_weaver/arc_engine/
pylint lattice_weaver/arc_engine/
```

### 3. Testear

```bash
# Ejecutar tests
pytest tests/unit/arc_engine/ -v

# Verificar cobertura
pytest --cov=lattice_weaver.arc_engine --cov-report=term-missing

# Si cobertura < 85%, añadir más tests
```

### 4. Documentar

```bash
# Generar documentación
python scripts/automation/generate_i18n.py

# Verificar que se generaron archivos i18n
ls docs/i18n/es/
ls docs/i18n/en/
ls docs/i18n/fr/
```

### 5. Commit y Push

```bash
# Añadir cambios
git add .

# Commit con mensaje convencional
git commit -m "feat(arc-engine): implementar AC-3.1 con last support

Añade optimización de last support al algoritmo AC-3.1.

- Implementa estructura de datos para last support
- Añade función revise_with_last_support()
- Actualiza tests unitarios
- Cobertura: 89%

Closes #42"

# Push a rama remota
git push origin feature/A-W2-T1-ac31-optimization
```

### 6. Crear Pull Request

```bash
# Crear PR usando GitHub CLI
gh pr create \
  --title "feat(arc-engine): implementar AC-3.1 con last support" \
  --body "Implementa optimización de last support para AC-3.1.

## Cambios
- Estructura de datos para last support
- Función revise_with_last_support()
- Tests unitarios actualizados

## Métricas
- Tests: 52/52 ✅
- Cobertura: 89%
- Lint: 0 errores

## Checklist
- [x] Tests pasan
- [x] Cobertura >= 85%
- [x] Documentación actualizada
- [x] Lint sin errores" \
  --base track-a-core
```

### 7. Actualizar Estado

```bash
# Marcar tarea como completada
python scripts/update_status.py \
  --agent-id agent-track-a \
  --task-id A-W2-T1 \
  --status COMPLETED

# Actualizar métricas
python scripts/update_metrics.py \
  --agent-id agent-track-a \
  --tests-passed 52 \
  --tests-total 52 \
  --coverage 89.0
```

---

## 🔍 Verificación Continua

### Verificación Pre-Commit

```bash
# Ejecutada automáticamente por pre-commit hook
black --check .
isort --check .
pylint lattice_weaver/
mypy lattice_weaver/
pytest tests/unit/ -q
```

### Verificación en CI/CD

**GitHub Actions:** `.github/workflows/track-a-ci.yml`

```yaml
name: Track A - CI

on:
  push:
    branches: [ track-a-core, feature/A-* ]
  pull_request:
    branches: [ track-a-core ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements-track-a.txt
        pip install pytest pytest-cov black isort pylint mypy
    
    - name: Lint
      run: |
        black --check lattice_weaver/arc_engine/
        isort --check lattice_weaver/arc_engine/
        pylint lattice_weaver/arc_engine/
        mypy lattice_weaver/arc_engine/
    
    - name: Test
      run: |
        pytest tests/unit/arc_engine/ --cov=lattice_weaver.arc_engine --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

---

## 📊 Reportes de Progreso

### Reporte Diario

```bash
# Generar reporte diario
python scripts/generate_daily_report.py --agent-id agent-track-a

# Enviar a GitHub (como issue comment)
python scripts/post_report_to_github.py --agent-id agent-track-a --report daily
```

**Formato del reporte:**

```markdown
## Reporte Diario - Agent Track A
**Fecha:** 2025-10-12
**Semana:** 2/8

### Progreso
- Tarea actual: A-W2-T1 - Implementar AC-3.1 optimizado
- Estado: IN_PROGRESS (80% completado)
- Tiempo invertido hoy: 6h

### Logros
- ✅ Implementada estructura de last support
- ✅ Añadida función revise_with_last_support()
- ✅ Tests unitarios: 52/52 pasando

### Próximos pasos
- Optimizar performance de last support
- Añadir benchmarks
- Documentar algoritmo

### Métricas
- Tests: 52/52 ✅
- Cobertura: 89%
- LOC: 2,450 (+110)
```

### Reporte Semanal

```bash
# Generar reporte semanal
python scripts/generate_weekly_report.py --agent-id agent-track-a --week 2
```

---

## 🚨 Manejo de Errores y Bloqueos

### Si hay conflictos en merge

```bash
# 1. Identificar archivos en conflicto
git status

# 2. Resolver manualmente
# Editar archivos con conflictos

# 3. Marcar como resueltos
git add <archivos-resueltos>

# 4. Completar merge
git commit

# 5. Actualizar estado
python scripts/update_status.py --agent-id agent-track-a --flag CONFLICT_RESOLVED
```

### Si tests fallan

```bash
# 1. Identificar tests fallidos
pytest tests/ -v --tb=short

# 2. Ejecutar solo tests fallidos
pytest tests/unit/arc_engine/test_core.py::test_enforce_arc_consistency -v

# 3. Debuggear
pytest tests/unit/arc_engine/test_core.py::test_enforce_arc_consistency --pdb

# 4. Corregir y re-testear
# ... fix code ...
pytest tests/unit/arc_engine/test_core.py::test_enforce_arc_consistency -v
```

### Si hay dependencias bloqueadas

```bash
# 1. Verificar estado de dependencias
python scripts/check_dependencies.py --agent-id agent-track-a

# 2. Si Track X está bloqueando, notificar
python scripts/notify_dependency_blocked.py \
  --agent-id agent-track-a \
  --blocked-by track-x \
  --reason "Esperando API de módulo Y"

# 3. Entrar en modo IDLE y buscar tareas alternativas
python scripts/enter_idle_mode.py --agent-id agent-track-a
```

---

## ✅ Checklist de Arranque

- [ ] Entorno virtual creado y activado
- [ ] Dependencias instaladas
- [ ] Sincronización con GitHub exitosa
- [ ] Estado del proyecto verificado
- [ ] Plan de desarrollo cargado
- [ ] Herramientas de desarrollo configuradas (black, isort, pylint, mypy)
- [ ] Pre-commit hooks instalados
- [ ] Tests ejecutados correctamente
- [ ] Documentación generada
- [ ] Estado del agente actualizado en GitHub

**Comando de verificación completa:**

```bash
python scripts/verify_agent_setup.py --agent-id agent-track-a
```

---

## 📚 Referencias

- [Plan de Track A](docs/tracks/TRACK_A_PLAN.md)
- [Protocolo de Ejecución Autónoma](PROTOCOLO_EJECUCION_AUTONOMA.md)
- [Meta-Principios de Diseño](docs/LatticeWeaver_Meta_Principios_Diseño_v3.md)
- [Análisis de Dependencias](Analisis_Dependencias_Tracks.md)


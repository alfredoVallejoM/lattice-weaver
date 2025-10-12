# Guía de Contribución - LatticeWeaver

**Versión:** 5.0.0  
**Fecha:** 12 de Octubre, 2025

---

## ¡Bienvenido!

Gracias por tu interés en contribuir a LatticeWeaver. Este documento te guiará a través del proceso de contribución.

---

## Tabla de Contenidos

1. [Código de Conducta](#código-de-conducta)
2. [Cómo Contribuir](#cómo-contribuir)
3. [Configuración del Entorno](#configuración-del-entorno)
4. [Estándares de Código](#estándares-de-código)
5. [Proceso de Pull Request](#proceso-de-pull-request)
6. [Sistema de Tracks](#sistema-de-tracks)
7. [Testing](#testing)
8. [Documentación](#documentación)

---

## Código de Conducta

LatticeWeaver se adhiere al [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Al participar, se espera que mantengas este código.

### Principios Básicos

- **Respeto**: Trata a todos con respeto y consideración
- **Inclusividad**: Fomenta un ambiente acogedor para todos
- **Colaboración**: Trabaja constructivamente con otros
- **Profesionalismo**: Mantén un tono profesional en todas las interacciones

---

## Cómo Contribuir

Hay muchas formas de contribuir a LatticeWeaver:

### 1. Reportar Bugs

Si encuentras un bug, por favor abre un [issue en GitHub](https://github.com/latticeweaver/lattice-weaver/issues) con:

- **Descripción clara** del problema
- **Pasos para reproducir** el bug
- **Comportamiento esperado** vs comportamiento actual
- **Versión** de LatticeWeaver y Python
- **Sistema operativo**
- **Logs** o mensajes de error relevantes

**Template de Bug Report:**

```markdown
## Descripción del Bug
[Descripción clara y concisa]

## Pasos para Reproducir
1. ...
2. ...
3. ...

## Comportamiento Esperado
[Qué esperabas que sucediera]

## Comportamiento Actual
[Qué sucedió realmente]

## Entorno
- LatticeWeaver version: 5.0.0
- Python version: 3.11.0
- OS: Ubuntu 22.04

## Logs
```
[Pegar logs aquí]
```
```

### 2. Sugerir Features

Para sugerir nuevas funcionalidades, abre un [feature request](https://github.com/latticeweaver/lattice-weaver/issues) con:

- **Descripción** de la funcionalidad
- **Casos de uso** que resolvería
- **Alternativas** consideradas
- **Mockups** o ejemplos (si aplica)

### 3. Contribuir Código

Para contribuir código:

1. **Fork** el repositorio
2. **Crea una rama** para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Haz tus cambios** siguiendo los estándares de código
4. **Escribe tests** para tu código
5. **Commit** tus cambios (`git commit -m 'feat: add AmazingFeature'`)
6. **Push** a tu rama (`git push origin feature/AmazingFeature`)
7. **Abre un Pull Request**

### 4. Mejorar Documentación

La documentación es crucial. Puedes contribuir:

- Corrigiendo typos
- Mejorando explicaciones
- Añadiendo ejemplos
- Traduciendo a otros idiomas

### 5. Investigar Fenómenos Multidisciplinares

LatticeWeaver busca mapear fenómenos de múltiples disciplinas. Puedes contribuir:

- **Investigación profunda** de un fenómeno (50-100 páginas)
- **Diseño de mapeo** a CSP/FCA/TDA (30-50 páginas)
- **Implementación** del modelo
- **Tutoriales** educativos

Ver [VISION_MULTIDISCIPLINAR.md](../../track-i-educational-multidisciplinary/VISION_MULTIDISCIPLINAR.md) para detalles.

---

## Configuración del Entorno

### Requisitos

- Python >= 3.11
- Git
- pip

### Instalación para Desarrollo

```bash
# 1. Fork y clonar el repositorio
git clone https://github.com/TU_USUARIO/lattice-weaver.git
cd lattice-weaver

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar en modo desarrollo
pip install -e ".[dev]"

# 4. Instalar pre-commit hooks
pip install pre-commit
pre-commit install

# 5. Verificar instalación
pytest
```

### Estructura del Proyecto

```
lattice-weaver/
├── lattice_weaver/          # Código fuente
│   ├── arc_engine/          # Motor CSP
│   ├── locales/             # Motor FCA
│   ├── topology/            # Motor TDA
│   ├── visualization/       # Visualización
│   └── ...
├── tests/                   # Tests
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
├── docs/                    # Documentación
├── examples/                # Ejemplos
└── scripts/                 # Scripts de automatización
```

---

## Estándares de Código

### Python Style Guide

Seguimos [PEP 8](https://pep8.org/) y [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

**Herramientas:**
- **black**: Formateo automático
- **isort**: Ordenamiento de imports
- **pylint**: Linting
- **mypy**: Type checking

### Formateo

```bash
# Formatear código
black lattice_weaver/

# Ordenar imports
isort lattice_weaver/

# Linting
pylint lattice_weaver/

# Type checking
mypy lattice_weaver/
```

### Convenciones de Nombres

- **Módulos y paquetes**: `snake_case`
- **Clases**: `PascalCase`
- **Funciones y métodos**: `snake_case`
- **Constantes**: `UPPER_SNAKE_CASE`
- **Variables privadas**: `_leading_underscore`

**Ejemplo:**

```python
# Constantes
MAX_ITERATIONS = 1000

# Clase
class AdaptiveConsistencyEngine:
    """Motor de consistencia adaptativa."""
    
    def __init__(self, algorithm='auto'):
        self._algorithm = algorithm  # Variable privada
        self.iterations = 0          # Variable pública
    
    def solve(self, problem):
        """Resuelve un problema CSP."""
        return self._apply_consistency(problem)
    
    def _apply_consistency(self, problem):
        """Método privado."""
        pass
```

### Docstrings

Usamos el estilo Google para docstrings.

**Ejemplo:**

```python
def add_constraint(var1: str, var2: str, constraint: Callable) -> None:
    """Añade una restricción entre dos variables.
    
    Args:
        var1: Nombre de la primera variable.
        var2: Nombre de la segunda variable.
        constraint: Función que evalúa la restricción.
    
    Raises:
        ValueError: Si alguna variable no existe.
        TypeError: Si constraint no es callable.
    
    Example:
        >>> engine.add_constraint("x", "y", lambda a, b: a != b)
    """
    pass
```

### Type Hints

Usa type hints para mejorar la legibilidad y permitir type checking.

```python
from typing import List, Dict, Optional, Callable

def solve(
    problem: CSPProblem,
    timeout: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """Resuelve un problema CSP."""
    pass
```

---

## Proceso de Pull Request

### Antes de Abrir un PR

1. **Asegúrate de que los tests pasan**:
   ```bash
   pytest
   ```

2. **Verifica el formateo**:
   ```bash
   black --check lattice_weaver/
   isort --check lattice_weaver/
   ```

3. **Ejecuta linting**:
   ```bash
   pylint lattice_weaver/
   ```

4. **Actualiza la documentación** si es necesario

### Abrir un PR

1. **Título descriptivo** siguiendo [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat: add new feature`
   - `fix: resolve bug in AC-3`
   - `docs: update API reference`
   - `test: add tests for FCA`
   - `refactor: improve performance`

2. **Descripción completa**:
   ```markdown
   ## Descripción
   [Descripción de los cambios]
   
   ## Motivación
   [Por qué estos cambios son necesarios]
   
   ## Cambios
   - [Cambio 1]
   - [Cambio 2]
   
   ## Tests
   - [x] Tests unitarios añadidos
   - [x] Tests de integración añadidos
   - [x] Todos los tests pasan
   
   ## Checklist
   - [x] Código formateado con black
   - [x] Imports ordenados con isort
   - [x] Linting pasado
   - [x] Documentación actualizada
   - [x] Type hints añadidos
   ```

3. **Asigna reviewers** apropiados

4. **Vincula issues** relacionados: `Closes #123`

### Revisión de Código

- Los PRs requieren al menos **1 aprobación** de un maintainer
- Responde a los comentarios de forma constructiva
- Haz los cambios solicitados
- Re-solicita revisión después de cambios

---

## Sistema de Tracks

LatticeWeaver se desarrolla mediante 9 tracks paralelos. Si quieres contribuir a un track específico:

### Tracks Disponibles

1. **Track A - Core Engine**: Motor CSP
2. **Track B - Locales y Frames**: Motor FCA
3. **Track C - Problem Families**: Familias de problemas
4. **Track D - Inference Engine**: Motor de inferencia
5. **Track E - Web Application**: Aplicación web
6. **Track F - Desktop App**: Aplicación desktop
7. **Track G - Editing Dinámica**: Editor de problemas
8. **Track H - Formal Math**: Problemas matemáticos formales
9. **Track I - Educational**: Visualización y educación multidisciplinar

### Contribuir a un Track

1. **Lee el protocolo del track**:
   ```bash
   cat track-X-nombre/PROTOCOLO_ARRANQUE_AGENTE_X.md
   ```

2. **Verifica el estado del track**:
   ```bash
   python scripts/check_track_status.py --track track-X-nombre
   ```

3. **Sincroniza con GitHub**:
   ```bash
   python scripts/sync_agent.py --track track-X-nombre
   ```

4. **Trabaja en tu feature** siguiendo el protocolo

5. **Commit y push** a tu rama del track

Ver [COORDINACION_TRACKS_V3_FINAL.md](../../COORDINACION_TRACKS_V3_FINAL.md) para detalles.

---

## Testing

### Ejecutar Tests

```bash
# Todos los tests
pytest

# Tests unitarios
pytest tests/unit/

# Tests de integración
pytest tests/integration/

# Con cobertura
pytest --cov=lattice_weaver --cov-report=html

# Tests específicos
pytest tests/unit/test_arc_engine.py::test_ac3_basic
```

### Escribir Tests

Usamos **pytest** para testing.

**Estructura:**

```python
import pytest
from lattice_weaver.arc_engine import AdaptiveConsistencyEngine

class TestAdaptiveConsistencyEngine:
    """Tests para AdaptiveConsistencyEngine."""
    
    def setup_method(self):
        """Setup antes de cada test."""
        self.engine = AdaptiveConsistencyEngine()
    
    def test_add_variable(self):
        """Test añadir variable."""
        self.engine.add_variable("x", [1, 2, 3])
        assert "x" in self.engine.variables
        assert self.engine.get_domain("x") == [1, 2, 3]
    
    def test_add_constraint(self):
        """Test añadir restricción."""
        self.engine.add_variable("x", [1, 2])
        self.engine.add_variable("y", [1, 2])
        self.engine.add_constraint("x", "y", lambda a, b: a != b)
        assert len(self.engine.constraints) == 1
    
    def test_solve_simple(self):
        """Test resolver problema simple."""
        self.engine.add_variable("x", [1, 2])
        self.engine.add_variable("y", [1, 2])
        self.engine.add_constraint("x", "y", lambda a, b: a != b)
        solution = self.engine.solve()
        assert solution is not None
        assert solution["x"] != solution["y"]
    
    @pytest.mark.parametrize("n", [4, 8, 12])
    def test_n_queens(self, n):
        """Test N-queens para diferentes tamaños."""
        # ... implementación ...
        pass
```

### Cobertura de Tests

Mantenemos **>85% de cobertura** de código.

```bash
# Generar reporte de cobertura
pytest --cov=lattice_weaver --cov-report=html

# Ver reporte
open htmlcov/index.html
```

---

## Documentación

### Documentar Código

- **Docstrings** para todos los módulos, clases y funciones públicas
- **Type hints** para todos los parámetros y retornos
- **Ejemplos** en docstrings cuando sea apropiado

### Documentar Features

Cuando añades una nueva feature, actualiza:

1. **API Reference**: `docs/api/API_REFERENCE.md`
2. **Tutorial**: Añade ejemplo en `docs/tutorials/`
3. **README**: Si es una feature mayor
4. **CHANGELOG**: Añade entrada

### Generar Documentación

Usamos **Sphinx** para generar documentación.

```bash
# Instalar Sphinx
pip install sphinx sphinx-rtd-theme

# Generar documentación
cd docs/
make html

# Ver documentación
open _build/html/index.html
```

---

## Preguntas Frecuentes

### ¿Cómo empiezo a contribuir?

1. Lee esta guía completa
2. Configura tu entorno de desarrollo
3. Busca issues etiquetados como `good first issue`
4. Comenta en el issue que quieres trabajar en él
5. Haz un fork y comienza a trabajar

### ¿Cuánto tiempo toma revisar un PR?

Generalmente **2-5 días hábiles**. PRs más complejos pueden tomar más tiempo.

### ¿Puedo trabajar en múltiples issues simultáneamente?

Sí, pero recomendamos enfocarte en uno a la vez para mantener PRs manejables.

### ¿Qué hago si mi PR es rechazado?

No te desanimes. Lee los comentarios, haz los cambios sugeridos, y vuelve a intentar. Todos los maintainers fueron contributors novatos alguna vez.

---

## Contacto

- **GitHub Issues**: https://github.com/latticeweaver/lattice-weaver/issues
- **Discord**: https://discord.gg/latticeweaver
- **Email**: team@latticeweaver.dev

---

**¡Gracias por contribuir a LatticeWeaver!** 🎉


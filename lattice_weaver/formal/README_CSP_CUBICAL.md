# Integración CSP ↔ Tipos Cúbicos

**Autor:** LatticeWeaver Team
**Fecha:** 12 de Octubre de 2025
**Versión:** 1.0

## 1. Resumen

Este módulo implementa la integración completa entre el motor de Satisfacción de Restricciones (CSP) y el sistema de tipos cúbicos de LatticeWeaver. Permite traducir problemas CSP a tipos cúbicos, verificar soluciones formalmente, analizar la topología del espacio de soluciones y explotar simetrías para optimizar la búsqueda.

## 2. Componentes Principales

| Componente | Descripción |
|:---|:---|
| **CubicalCSPType** | Representa un problema CSP como un tipo cúbico (Sigma-Type). |
| **CSPToCubicalBridge** | Puente bidireccional entre ArcEngine y el sistema cúbico. |
| **PathFinder** | Busca caminos entre soluciones en el espacio de tipos. |
| **SymmetryExtractor** | Detecta y analiza simetrías en el problema CSP. |

## 3. Flujo de Trabajo

1. **Crear un CSP** usando `ArcEngine`.
2. **Instanciar `CSPToCubicalBridge`** con el motor CSP.
3. **Traducir** el CSP a un tipo cúbico.
4. **Verificar soluciones** usando el bridge.
5. **Analizar caminos** entre soluciones con `PathFinder`.
6. **Extraer simetrías** con `SymmetryExtractor`.

## 4. Ejemplo de Uso

```python
from lattice_weaver.formal.csp_cubical_bridge import create_simple_csp_bridge
from lattice_weaver.formal.path_finder import PathFinder

# Crear bridge
bridge = create_simple_csp_bridge(
    variables=["X", "Y"],
    domains={"X": {1, 2}, "Y": {1, 2}},
    constraints=[("X", "Y", lambda x, y: x != y)]
)

# Verificar solución
print(bridge.verify_solution({"X": 1, "Y": 2}))  # True

# Buscar camino
finder = PathFinder(bridge)
path = finder.find_path({"X": 1, "Y": 2}, {"X": 2, "Y": 1})
print(path)
```

## 5. Tests

Este módulo está cubierto por **116 tests** (102 unitarios, 14 de integración), con una cobertura del ~95%.

Para ejecutar los tests:

```bash
python3.11 -m pytest tests/unit/test_cubical_csp_type.py
python3.11 -m pytest tests/unit/test_csp_cubical_bridge.py
python3.11 -m pytest tests/unit/test_path_finder.py
python3.11 -m pytest tests/unit/test_symmetry_extractor.py
python3.11 -m pytest tests/integration/test_csp_cubical_integration.py
```


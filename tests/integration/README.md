# Tests de Integración - LatticeWeaver

## Descripción

Los tests de integración validan flujos completos a través de múltiples capas del sistema, asegurando que los componentes funcionan correctamente cuando se integran.

## Estructura

- `test_csp_to_hott_flow.py` - Flujo CSP → HoTT → Verificación (3 tests)
- `test_fca_to_topology_flow.py` - Flujo FCA → Topología → Homotopía (2 tests)
- `test_optimization_pipeline.py` - Pipeline de optimizaciones (3 tests)

**Total:** 8 tests de integración

## Ejecutar Tests

### Todos los tests de integración

```bash
pytest tests/integration -v
```

### Test específico

```bash
pytest tests/integration/test_csp_to_hott_flow.py -v
```

### Solo tests rápidos (excluir slow)

```bash
pytest tests/integration -v -m "not slow"
```

### Con cobertura

```bash
pytest tests/integration --cov=lattice_weaver --cov-report=html
```

### Usando el script automatizado

```bash
./scripts/run_tests.sh
```

## Markers Disponibles

- `@pytest.mark.integration` - Test de integración (todos los tests de este directorio)
- `@pytest.mark.slow` - Test lento (>1s de ejecución)

## Fixtures Disponibles

Ver `conftest.py` para fixtures disponibles:

### Motores Pre-configurados

- `arc_engine` - Motor CSP configurado
- `fca_engine` - Motor FCA configurado
- `tda_engine` - Motor TDA configurado
- `cubical_engine` - Motor HoTT configurado

### Problemas de Test

- `nqueens_problem` - Problema N-Reinas (n=4, 2 soluciones)
- `sudoku_problem` - Problema Sudoku 4x4 simplificado
- `graph_coloring_problem` - Problema de coloreo (5 nodos, 3 colores)
- `sample_lattice` - Retículo de conceptos pre-computado

## Ejemplos de Uso

### Usar fixture en un test

```python
@pytest.mark.integration
def test_my_integration(arc_engine, nqueens_problem):
    solutions = arc_engine.solve(nqueens_problem)
    assert len(solutions) > 0
```

### Marcar test como lento

```python
@pytest.mark.slow
def test_large_problem(arc_engine):
    # Test que toma >1s
    pass
```

## Convenciones

1. **Nombrar tests descriptivamente:** `test_<componente>_<accion>_<resultado>`
2. **Documentar flujos:** Incluir docstring explicando el flujo completo
3. **Assertions claras:** Mensajes descriptivos en assertions
4. **Marcar tests lentos:** Usar `@pytest.mark.slow` para tests >1s

## Agregar Nuevos Tests

1. Crear archivo `test_<nombre>.py` en este directorio
2. Importar fixtures necesarias de `conftest.py`
3. Marcar clase o función con `@pytest.mark.integration`
4. Documentar el flujo en docstring
5. Ejecutar para verificar: `pytest tests/integration/test_<nombre>.py -v`


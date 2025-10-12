# Truth Maintenance System (TMS)

**Estado:** COMPLETADO ✅  
**Fecha:** 11 de Octubre de 2025  
**Versión:** 1.0

---

## Resumen

Implementación de un **Truth Maintenance System (TMS)** que rastrea dependencias entre decisiones durante la resolución de CSP, permitiendo retroceso eficiente, explicación de inconsistencias y restauración incremental de consistencia.

---

## Archivos Implementados

### 1. `lattice_weaver/arc_engine/tms.py`

**Clases principales:**
- `Justification` - Justificación de eliminación de valor
- `Decision` - Decisión de asignación
- `TruthMaintenanceSystem` - Sistema principal de rastreo

**Funcionalidades:**
- Rastreo de dependencias entre decisiones
- Explicación de inconsistencias
- Sugerencia de restricciones a relajar
- Identificación de valores restaurables
- Grafo de conflictos
- Estadísticas

---

## Conceptos Clave

### Justification (Justificación)

Una **justificación** registra por qué se eliminó un valor del dominio de una variable.

```python
@dataclass
class Justification:
    variable: str                    # Variable afectada
    removed_value: Any               # Valor eliminado
    reason_constraint: str           # Restricción que causó la eliminación
    supporting_values: Dict[str, Any]  # Valores que justifican esto
```

**Ejemplo:**
```
Variable: X
Valor eliminado: 1
Restricción: C1 (X != Y)
Valores de soporte: {Y: [1]}
Explicación: X=1 fue eliminado porque Y=1 y C1 requiere X != Y
```

### Decision (Decisión)

Una **decisión** registra una asignación de variable durante la búsqueda.

```python
@dataclass
class Decision:
    variable: str
    value: Any
    justifications: List[Justification]  # Justificaciones que dependen de esta decisión
```

---

## Funcionalidades del TMS

### 1. Rastreo de Eliminaciones

El TMS registra cada eliminación de valor durante AC-3:

```python
tms = create_tms()

# Durante AC-3, cuando se elimina un valor:
tms.record_removal(
    variable="X",
    value=1,
    constraint_id="C1",
    supporting_values={"Y": [2, 3]}
)
```

**Internamente:**
- Crea una `Justification`
- La agrega al grafo de dependencias
- La indexa por restricción

### 2. Explicación de Inconsistencias

Cuando una variable queda sin valores, el TMS explica por qué:

```python
# Variable X quedó vacía
explanations = tms.explain_inconsistency("X")

for exp in explanations:
    print(f"Valor {exp.removed_value} eliminado por {exp.reason_constraint}")
```

**Salida ejemplo:**
```
Valor 1 eliminado por C1
Valor 2 eliminado por C2
Valor 3 eliminado por C1
```

### 3. Sugerencia de Restricción a Relajar

El TMS sugiere qué restricción relajar para resolver la inconsistencia:

```python
suggested = tms.suggest_constraint_to_relax("X")
print(f"Sugerencia: relajar restricción '{suggested}'")
```

**Estrategia:** Restricción que causó más eliminaciones.

### 4. Identificación de Valores Restaurables

Al eliminar una restricción, el TMS identifica qué valores pueden restaurarse:

```python
restorable = tms.get_restorable_values("C1")

# Ejemplo: {'X': [1, 2], 'Y': [3]}
```

### 5. Grafo de Conflictos

El TMS construye un grafo de conflictos mostrando qué variables están involucradas:

```python
conflict_graph = tms.get_conflict_graph("X")

# Ejemplo: {'C1': ['Y', 'Z'], 'C2': ['Y']}
```

### 6. Estadísticas

```python
stats = tms.get_statistics()

# Ejemplo:
# {
#     'total_justifications': 10,
#     'total_decisions': 5,
#     'variables_with_removals': 3,
#     'constraints_involved': 2,
#     'avg_removals_per_variable': 3.33
# }
```

---

## Integración con ArcEngine

### Habilitación del TMS

```python
from lattice_weaver.arc_engine import ArcEngine

# Crear engine con TMS habilitado
engine = ArcEngine(use_tms=True)

# El TMS se activa automáticamente
assert engine.tms is not None
```

### Rastreo Automático

Durante `enforce_arc_consistency()`, el TMS registra automáticamente:

```python
engine.add_variable("X", [1, 2, 3])
engine.add_variable("Y", [1, 2, 3])
engine.add_constraint("X", "Y", lambda x, y: x != y, cid="C1")

# AC-3 con rastreo TMS
consistent = engine.enforce_arc_consistency()

# Ver justificaciones
print(f"Justificaciones: {len(engine.tms.justifications)}")
```

### Explicación de Fallos

Si AC-3 detecta inconsistencia, el TMS explica automáticamente:

```python
# Problema inconsistente
engine.add_variable("X", [1])
engine.add_variable("Y", [2])
engine.add_constraint("X", "Y", lambda x, y: x == y, cid="C1")

consistent = engine.enforce_arc_consistency()

# Salida automática:
# ⚠️ Sugerencia: relajar restricción 'C1'
```

### Eliminación de Restricciones con Restauración

El TMS permite eliminar restricciones y restaurar valores eficientemente:

```python
# Problema: X < Y < Z
engine.add_variable("X", [1, 2, 3])
engine.add_variable("Y", [1, 2, 3])
engine.add_variable("Z", [1, 2, 3])
engine.add_constraint("X", "Y", lambda x, y: x < y, cid="C1")
engine.add_constraint("Y", "Z", lambda y, z: y < z, cid="C2")

# AC-3 reduce dominios
engine.enforce_arc_consistency()

# Eliminar C1 - valores restaurados automáticamente
engine.remove_constraint("C1")

# Salida:
# ✅ Restaurado: X=3
# Restricción 'C1' eliminada
```

---

## Casos de Uso

### 1. Debugging de CSP

Entender por qué un problema es inconsistente:

```python
engine = ArcEngine(use_tms=True)

# Definir problema
# ...

if not engine.enforce_arc_consistency():
    # Analizar fallo
    for var in engine.variables:
        if not engine.variables[var]:
            explanations = engine.tms.explain_inconsistency(var)
            print(f"\nVariable {var} inconsistente:")
            for exp in explanations:
                print(f"  {exp.removed_value} eliminado por {exp.reason_constraint}")
```

### 2. Relajación de Restricciones

Resolver problemas sobre-restringidos:

```python
engine = ArcEngine(use_tms=True)

# Problema sobre-restringido
# ...

if not engine.enforce_arc_consistency():
    # Identificar restricción problemática
    for var in engine.variables:
        if not engine.variables[var]:
            suggested = engine.tms.suggest_constraint_to_relax(var)
            print(f"Relajar: {suggested}")
            
            # Eliminar restricción
            engine.remove_constraint(suggested)
            
            # Reintentar
            if engine.enforce_arc_consistency():
                print("Problema resuelto!")
```

### 3. Modificación Dinámica de CSP

Agregar/eliminar restricciones dinámicamente:

```python
engine = ArcEngine(use_tms=True)

# Problema inicial
# ...
engine.enforce_arc_consistency()

# Agregar restricción temporal
engine.add_constraint("X", "Y", lambda x, y: x > y, cid="C_temp")
engine.enforce_arc_consistency()

# Eliminar restricción temporal - valores restaurados
engine.remove_constraint("C_temp")
```

### 4. Análisis de Conflictos

Identificar grupos de restricciones conflictivas:

```python
engine = ArcEngine(use_tms=True)

# Problema
# ...

if not engine.enforce_arc_consistency():
    for var in engine.variables:
        if not engine.variables[var]:
            conflict_graph = engine.tms.get_conflict_graph(var)
            
            print(f"\nConflictos en {var}:")
            for constraint, involved_vars in conflict_graph.items():
                print(f"  {constraint}: {involved_vars}")
```

---

## Tests Implementados

### `tests/test_tms.py`

**9 tests completos:**

1. ✅ **Test 1:** Funcionalidad básica del TMS
2. ✅ **Test 2:** Explicación de inconsistencias
3. ✅ **Test 3:** Sugerencia de restricción a relajar
4. ✅ **Test 4:** Valores restaurables
5. ✅ **Test 5:** Integración TMS con ArcEngine
6. ✅ **Test 6:** Eliminación de restricción con restauración
7. ✅ **Test 7:** Grafo de conflictos
8. ✅ **Test 8:** Estadísticas del TMS
9. ✅ **Test 9:** Limpieza del TMS

**Resultado:** 9/9 tests pasados ✅

---

## Arquitectura Interna

### Estructuras de Datos

```python
class TruthMaintenanceSystem:
    justifications: List[Justification]          # Lista de todas las justificaciones
    decisions: List[Decision]                    # Lista de decisiones
    dependency_graph: Dict[str, Set[Justification]]  # {variable: justificaciones}
    constraint_removals: Dict[str, List[Justification]]  # {constraint_id: justificaciones}
```

### Flujo de Rastreo

1. **Durante AC-3:**
   - `revise()` elimina valor
   - `record_removal()` crea Justification
   - Justification se agrega a `dependency_graph` y `constraint_removals`

2. **Detección de inconsistencia:**
   - Variable queda vacía
   - `explain_inconsistency()` consulta `dependency_graph`
   - `suggest_constraint_to_relax()` cuenta eliminaciones por restricción

3. **Eliminación de restricción:**
   - `get_restorable_values()` consulta `constraint_removals`
   - Para cada valor, verifica consistencia con restricciones restantes
   - Restaura valores consistentes
   - `remove_constraint_justifications()` limpia TMS

---

## Complejidad

### Rastreo de Eliminaciones

- **Tiempo:** O(1) por eliminación
- **Espacio:** O(k × d) donde k = restricciones, d = tamaño promedio de dominio

### Explicación de Inconsistencias

- **Tiempo:** O(e) donde e = eliminaciones en variable
- **Espacio:** O(1)

### Sugerencia de Restricción

- **Tiempo:** O(e) donde e = eliminaciones en variable
- **Espacio:** O(k) donde k = restricciones

### Restauración de Valores

- **Tiempo:** O(v × k × d) donde v = valores a restaurar, k = restricciones, d = tamaño de dominio
- **Espacio:** O(v)

---

## Beneficios

### 1. Debugging Eficiente

- **Explicación clara** de por qué falló el problema
- **Identificación rápida** de restricciones problemáticas
- **Visualización** de dependencias

### 2. Retroceso Eficiente

- **O(v × k × d)** en lugar de recomputar todo
- **Restauración incremental** de consistencia
- **Modificación dinámica** de problemas

### 3. Análisis de Conflictos

- **Grafo de conflictos** para visualización
- **Identificación de grupos** de restricciones relacionadas
- **Base para aprendizaje** de patrones de conflicto

### 4. Flexibilidad

- **Activación opcional** (flag `use_tms`)
- **Overhead mínimo** cuando está deshabilitado
- **Integración transparente** con AC-3

---

## Limitaciones

### 1. Overhead de Memoria

- Rastrea todas las eliminaciones: O(k × d) espacio
- Para problemas muy grandes, puede ser significativo
- Solución: Limitar profundidad de rastreo

### 2. Restauración No Garantizada

- Valores restaurados deben ser consistentes con restricciones restantes
- No todos los valores eliminados son restaurables
- Puede requerir re-ejecución de AC-3

### 3. No Rastrea Decisiones de Búsqueda

- Actualmente solo rastrea AC-3
- No integrado con backtracking search
- Solución futura: Extender a búsqueda completa

---

## Mejoras Futuras

### 1. Integración con Búsqueda

- Rastrear decisiones de backtracking
- Retroceso dirigido (conflict-directed backjumping)
- Aprendizaje de nogood clauses

### 2. Optimización de Memoria

- Limitar profundidad de rastreo
- Comprimir justificaciones antiguas
- Garbage collection de justificaciones no usadas

### 3. Análisis Avanzado

- Minimal Conflict Sets (MCS)
- Maximal Satisfiable Subsets (MSS)
- Explicaciones mínimas

### 4. Visualización

- Grafo de dependencias interactivo
- Timeline de eliminaciones
- Heatmap de restricciones conflictivas

---

## Comparación con Otros Sistemas

### TMS Clásico (Doyle, 1979)

**Similitudes:**
- Rastreo de dependencias
- Justificaciones

**Diferencias:**
- LatticeWeaver enfocado en CSP
- Más simple y eficiente
- No soporta creencias no-monotónicas

### ATMS (de Kleer, 1986)

**Similitudes:**
- Rastreo de suposiciones
- Análisis de conflictos

**Diferencias:**
- ATMS más general
- LatticeWeaver optimizado para CSP
- Menor overhead

### Sistemas CSP Modernos

**Similitudes:**
- Explicación de fallos
- Análisis de conflictos

**Diferencias:**
- LatticeWeaver integrado con HoTT
- Verificación formal disponible
- Arquitectura modular

---

## Conclusión

El Truth Maintenance System implementa exitosamente:

- ✅ Rastreo de dependencias eficiente
- ✅ Explicación de inconsistencias
- ✅ Sugerencia de restricciones a relajar
- ✅ Restauración incremental de consistencia
- ✅ Análisis de conflictos
- ✅ Integración transparente con ArcEngine
- ✅ Tests validados (9/9 ✅)

**Próxima fase:** Refactorización para Paralelización Real

---

## Referencias

- **TMS:** Doyle, J. (1979). "A Truth Maintenance System"
- **ATMS:** de Kleer, J. (1986). "An Assumption-Based TMS"
- **CSP:** Dechter, R. "Constraint Processing"
- **Conflict Analysis:** Ginsberg, M. "Dynamic Backtracking"


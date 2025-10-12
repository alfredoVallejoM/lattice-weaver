# Tácticas Avanzadas de Búsqueda de Pruebas

**Estado:** COMPLETADO ✅  
**Fecha:** 11 de Octubre de 2025  
**Versión:** 1.0

---

## Resumen

Implementación de **tácticas avanzadas de búsqueda de pruebas** inspiradas en asistentes de pruebas como Coq y Lean, que permiten búsqueda automática de pruebas más complejas en el sistema formal HoTT.

---

## Archivos Implementados

### 1. `lattice_weaver/formal/tactics.py`

**Clase principal:** `TacticEngine`

**Funcionalidades:**
- Motor de tácticas con estrategias avanzadas
- Tácticas básicas mejoradas
- Táctica automática con recursión
- Estadísticas de uso

**Tácticas implementadas:**

```python
class TacticEngine:
    def apply_reflexivity(self, goal) -> TacticResult
    def apply_assumption(self, goal) -> TacticResult
    def apply_intro(self, goal, var_name=None) -> TacticResult
    def apply_split(self, goal) -> TacticResult
    def apply_contradiction(self, goal) -> TacticResult
    def apply_rewrite(self, goal, equality_var) -> TacticResult
    def apply_auto(self, goal, max_depth=3) -> TacticResult
```

### 2. Extensión de `CubicalEngine`

**Método nuevo:** `search_proof_with_tactics`

Integra las tácticas avanzadas con el motor de búsqueda existente.

---

## Tácticas Implementadas

### 1. Reflexivity

**Propósito:** Probar igualdades reflexivas `a = a`

**Uso:**
```python
# Meta: x = x
result = tactics.apply_reflexivity(goal)
# Genera: refl x
```

**Aplicable a:** Tipos Path donde left = right

---

### 2. Assumption

**Propósito:** Usar una hipótesis del contexto

**Uso:**
```python
# Contexto: h : A
# Meta: A
result = tactics.apply_assumption(goal)
# Genera: h
```

**Aplicable a:** Cualquier meta que coincida con una hipótesis

---

### 3. Intro

**Propósito:** Introducir una variable para probar funciones

**Uso:**
```python
# Meta: A → B
result = tactics.apply_intro(goal)
# Genera submeta: B (con x : A en contexto)
```

**Aplicable a:** Tipos Pi (funciones)

**Resultado:** Submeta con contexto extendido

---

### 4. Split

**Propósito:** Dividir productos en componentes

**Uso:**
```python
# Meta: A × B
result = tactics.apply_split(goal)
# Genera 2 submetas: A y B
```

**Aplicable a:** Tipos Sigma no dependientes (productos)

**Resultado:** Dos submetas independientes

---

### 5. Contradiction

**Propósito:** Probar cualquier cosa desde falsedad (⊥)

**Uso:**
```python
# Contexto: absurd_hyp : ⊥
# Meta: A (cualquier cosa)
result = tactics.apply_contradiction(goal)
# Genera: (absurd absurd_hyp)
```

**Aplicable a:** Contextos con tipo Empty/⊥

---

### 6. Rewrite

**Propósito:** Reescribir usando una igualdad

**Uso:**
```python
# Contexto: eq : a = b
# Meta: P a
result = tactics.apply_rewrite(goal, "eq")
# Genera submeta: P b
```

**Aplicable a:** Contextos con igualdades (Path)

**Nota:** Implementación simplificada

---

### 7. Auto (Táctica Automática)

**Propósito:** Intentar resolver automáticamente la meta

**Estrategia:**
1. Intentar tácticas directas (reflexivity, assumption, contradiction)
2. Si falla, intentar intro + recursión
3. Si falla, intentar split + recursión en ambas submetas

**Uso:**
```python
# Meta: A → A
result = tactics.apply_auto(goal, max_depth=3)
# Genera: λ(x : A). x
```

**Aplicable a:** Metas simples que se pueden resolver automáticamente

**Profundidad:** Controla cuántas veces se puede recurrir

---

## Resultado de Tácticas

Todas las tácticas retornan un `TacticResult`:

```python
@dataclass
class TacticResult:
    success: bool                    # Si tuvo éxito
    proof: Optional[ProofTerm]       # Prueba generada (si completa)
    subgoals: List[ProofGoal]        # Submetas generadas
    message: str                     # Mensaje descriptivo
```

**Casos:**
- **Prueba completa:** `success=True`, `proof` presente
- **Submetas generadas:** `success=True`, `subgoals` presente
- **Fallo:** `success=False`, `message` explica por qué

---

## Integración con CubicalEngine

### Uso Básico

```python
from lattice_weaver.formal import CubicalEngine, ProofGoal

engine = CubicalEngine()

# Acceder a tácticas
tactics = engine.tactics

# Aplicar táctica
result = tactics.apply_auto(goal)

if result.success and result.proof:
    print(f"Prueba: {result.proof.term}")
```

### Búsqueda con Tácticas

```python
# Búsqueda automática con tácticas avanzadas
proof = engine.search_proof_with_tactics(
    goal,
    max_depth=5,
    use_advanced_tactics=True
)
```

**Comportamiento:**
1. Intenta búsqueda básica primero
2. Si falla, usa táctica `auto`
3. Retorna prueba o None

---

## Tests Implementados

### `tests/test_tactics.py`

**10 tests completos:**

1. ✅ **Test 1:** Táctica de reflexividad
2. ✅ **Test 2:** Táctica de asunción
3. ✅ **Test 3:** Táctica de introducción
4. ✅ **Test 4:** Táctica de división (split)
5. ✅ **Test 5:** Táctica de contradicción
6. ✅ **Test 6:** Táctica automática (simple)
7. ✅ **Test 7:** Táctica automática (con intro)
8. ✅ **Test 8:** Táctica automática (con split)
9. ✅ **Test 9:** Estadísticas de tácticas
10. ✅ **Test 10:** Búsqueda con tácticas integrada

**Resultado:** 10/10 tests pasados ✅

---

## Ejemplos de Uso

### Ejemplo 1: Función Identidad

```python
# Meta: A → A
goal = ProofGoal(
    PiType("x", TypeVar("A"), TypeVar("A")),
    ctx,
    "identity"
)

proof = engine.search_proof_with_tactics(goal)
# Resultado: λ(x : A). x
```

### Ejemplo 2: Función Constante

```python
# Meta: A → B → A
goal = ProofGoal(
    PiType("x", TypeVar("A"), 
           PiType("y", TypeVar("B"), TypeVar("A"))),
    ctx,
    "const"
)

proof = engine.search_proof_with_tactics(goal)
# Resultado: λ(x : A). λ(y : B). x
```

### Ejemplo 3: Construcción de Par

```python
# Contexto: a : A, b : B
# Meta: A × B
goal = ProofGoal(
    SigmaType("_", TypeVar("A"), TypeVar("B")),
    ctx,
    "pair"
)

proof = engine.search_proof_with_tactics(goal)
# Resultado: (a, b)
```

### Ejemplo 4: Ex Falso Quodlibet

```python
# Contexto: absurd_proof : ⊥
# Meta: A
goal = ProofGoal(TypeVar("A"), ctx, "from_false")

proof = engine.search_proof_with_tactics(goal)
# Resultado: (absurd absurd_proof)
```

---

## Estadísticas

El motor de tácticas rastrea el uso de cada táctica:

```python
# Obtener estadísticas
stats = tactics.get_statistics()

# Ejemplo de salida:
# {
#     'reflexivity': 5,
#     'assumption': 3,
#     'intro': 2,
#     'split': 1,
#     'auto': 4,
#     ...
# }

# Resetear
tactics.reset_statistics()
```

**Uso:** Analizar qué tácticas son más efectivas

---

## Arquitectura

### Flujo de Búsqueda con Tácticas

```
search_proof_with_tactics(goal)
    ↓
search_proof(goal)  [búsqueda básica]
    ↓ (si falla)
apply_auto(goal)
    ↓
Intentar tácticas directas:
    - reflexivity
    - assumption
    - contradiction
    ↓ (si falla)
Intentar intro + recursión
    ↓ (si falla)
Intentar split + recursión
    ↓
Retornar resultado
```

### Táctica Auto (Detalle)

```
apply_auto(goal, depth)
    ↓
¿Es Path con left=right? → reflexivity
    ↓
¿Está en contexto? → assumption
    ↓
¿Hay ⊥ en contexto? → contradiction
    ↓
¿Es Pi? → intro + auto(submeta, depth-1)
    ↓
¿Es Sigma? → split + auto(sub1) + auto(sub2)
    ↓
Fallo
```

---

## Beneficios

### 1. Mayor Poder de Prueba

Capacidad de probar proposiciones más complejas automáticamente:
- Funciones anidadas (A → B → C)
- Productos (A × B × C)
- Combinaciones de ambos

### 2. Automatización

Menos intervención manual:
- Táctica `auto` resuelve casos comunes
- Recursión automática en submetas
- Estrategias inteligentes

### 3. Flexibilidad

Adaptable a diferentes tipos de problemas:
- Tácticas específicas para casos especiales
- Combinación de tácticas
- Control de profundidad

### 4. Debugging

Estadísticas y mensajes descriptivos:
- Rastreo de tácticas usadas
- Mensajes de error claros
- Submetas explícitas

---

## Limitaciones

### 1. Tácticas Simplificadas

Algunas tácticas son versiones simplificadas:
- **Rewrite:** No implementa transport completo
- **Inducción:** No implementada (requiere tipos inductivos)
- **Casos:** No implementada (requiere tipos suma)

### 2. Profundidad Limitada

La recursión tiene límite de profundidad:
- Evita loops infinitos
- Puede fallar en pruebas complejas
- Requiere ajuste manual de `max_depth`

### 3. No Backtracking

No hay backtracking inteligente:
- Si una táctica falla, no se prueban alternativas
- Orden de tácticas es fijo
- Puede perder soluciones

---

## Mejoras Futuras

### 1. Tácticas Adicionales

- **Inducción:** Para tipos Nat, List, etc.
- **Casos:** Para tipos suma (A + B)
- **Rewrite completo:** Con transport
- **Simplificación:** Normalización de términos

### 2. Heurísticas Inteligentes

- **Backtracking:** Probar alternativas
- **Priorización:** Ordenar tácticas por probabilidad
- **Aprendizaje:** Adaptar estrategias según éxito

### 3. Integración con SMT

- **Solver externo:** Para lógica proposicional
- **Verificación:** Validar pruebas generadas
- **Optimización:** Usar SMT para submetas

### 4. Tácticas de Usuario

- **Definición:** Permitir definir tácticas personalizadas
- **Composición:** Combinar tácticas existentes
- **Macros:** Secuencias de tácticas frecuentes

---

## Comparación con Asistentes de Pruebas

### Coq

**Similitudes:**
- Tácticas básicas (intro, split, reflexivity)
- Táctica auto
- Submetas

**Diferencias:**
- Coq tiene muchas más tácticas
- Coq tiene Ltac (lenguaje de tácticas)
- Coq tiene backtracking

### Lean

**Similitudes:**
- Tácticas básicas
- Búsqueda automática
- Estadísticas

**Diferencias:**
- Lean tiene tactic mode interactivo
- Lean tiene simp (simplificación)
- Lean tiene omega (aritmética)

### Agda

**Similitudes:**
- Reflexividad
- Introducción

**Diferencias:**
- Agda es más interactivo (holes)
- Agda tiene pattern matching
- Agda tiene auto (más potente)

---

## Conclusión

Las tácticas avanzadas implementan exitosamente:

- ✅ 7 tácticas básicas y avanzadas
- ✅ Táctica automática con recursión
- ✅ Integración con CubicalEngine
- ✅ Estadísticas de uso
- ✅ Tests validados (10/10 ✅)
- ✅ Ejemplos completos

**Próxima fase:** Verificación Formal de Propiedades CSP

---

## Referencias

- **Coq Reference Manual:** https://coq.inria.fr/refman/
- **Lean Tactics:** https://leanprover.github.io/theorem_proving_in_lean/
- **Agda Documentation:** https://agda.readthedocs.io/
- **HoTT Book:** https://homotopytypetheory.org/book/


# Análisis de Compatibilidad Actualizado: Solver Actual vs ArcEngine

**Fecha:** 16 de Octubre, 2025  
**Actualización:** Tras verificar que ArcEngine existe en rama `feature/fibration-flow-core-refactor`

---

## 🔍 Hallazgos Clave

### 1. Estado Real del ArcEngine

Tras revisar el repositorio y las ramas:

**En rama `main`:**
- ✅ `CSPSolver` implementado en `core/csp_engine/solver.py`
- ✅ AC-3 básico funcional
- ✅ Backtracking con forward checking
- ❌ **NO existe ArcEngine/AdaptiveConsistencyEngine**

**En rama `feature/fibration-flow-core-refactor`:**
- ⚠️ Se **importa** `AdaptiveConsistencyEngine` como `ArcEngine` en `adaptive/phase0.py`
- ❌ Pero la clase **NO está definida** en `core/csp_engine/solver.py`
- ⚠️ El import falla: `from lattice_weaver.core.csp_engine.solver import AdaptiveConsistencyEngine`

**Conclusión:** El ArcEngine está **referenciado pero no implementado** en ninguna rama actual del repositorio.

### 2. Lo Que Existe vs Lo Que Se Espera

**Existe:**
```python
# lattice_weaver/core/csp_engine/solver.py (main)
class CSPSolver:
    """Solver básico con backtracking + forward checking + AC-3"""
    def enforce_arc_consistency(self) -> bool:
        # AC-3 básico implementado
```

**Se espera (según documentación y código en adaptive/phase0.py):**
```python
# Debería existir pero NO existe
class AdaptiveConsistencyEngine:
    """Motor adaptativo con AC-3.1, clustering, optimizaciones"""
    # Referenciado en múltiples lugares pero no implementado
```

---

## 🎯 Situación Real: Dos Caminos Posibles

### Opción A: El ArcEngine Existe en Otra Ubicación/Rama

**Posibilidad:** El ArcEngine podría estar en:
1. Una rama no sincronizada
2. Un commit antiguo
3. Archivos locales no subidos
4. Otra estructura de carpetas

**Acción necesaria:** Verificar con el equipo dónde está la implementación real del ArcEngine.

### Opción B: El ArcEngine Nunca Se Implementó Completamente

**Posibilidad:** 
- Se diseñó la arquitectura (documentación extensa)
- Se crearon referencias en el código (imports)
- Pero la implementación real **nunca se completó**

**Evidencia:**
- El `adaptive/phase0.py` intenta importarlo pero fallaría en ejecución
- La documentación lo menciona extensamente
- Pero no hay código fuente de la clase

---

## 📊 Análisis de Impacto: ¿Qué Hacer?

### Escenario 1: Si el ArcEngine Existe en Algún Lugar

**Estrategia:** Merge cuidadoso con el CSPSolver actual

1. **Evaluar compatibilidad de APIs**
   ```python
   # API del CSPSolver actual
   solver = CSPSolver(csp, tracer)
   stats = solver.solve()
   
   # API esperada del ArcEngine (según adaptive/phase0.py)
   engine = ArcEngine()
   engine.add_variable(name, domain)
   engine.add_constraint(var1, var2, relation)
   solution = engine.solve()
   ```
   
   **Observación:** Las APIs son **incompatibles**
   - CSPSolver recibe un objeto CSP completo
   - ArcEngine construye el problema incrementalmente

2. **Estrategia de integración: Adapter Pattern**
   ```python
   class ArcEngineAdapter:
       """Adapta ArcEngine para usar la API de CSPSolver"""
       
       def __init__(self, csp: CSP, tracer=None):
           self.arc_engine = ArcEngine()
           self._populate_from_csp(csp)
           self.tracer = tracer
       
       def _populate_from_csp(self, csp: CSP):
           """Convierte CSP a formato de ArcEngine"""
           for var in csp.variables:
               self.arc_engine.add_variable(var, csp.domains[var])
           
           for constraint in csp.constraints:
               if len(constraint.scope) == 2:
                   var1, var2 = constraint.scope
                   self.arc_engine.add_constraint(var1, var2, constraint.relation)
       
       def solve(self, all_solutions=False, max_solutions=1):
           """API compatible con CSPSolver"""
           solution = self.arc_engine.solve()
           
           # Convertir a formato CSPSolutionStats
           stats = CSPSolutionStats()
           if solution:
               stats.solutions.append(CSPSolution(assignment=solution))
           return stats
   ```

3. **Uso unificado**
   ```python
   # Código existente sigue funcionando
   solver = CSPSolver(csp, tracer)
   stats = solver.solve()
   
   # Nuevo código puede usar ArcEngine con misma API
   solver = ArcEngineAdapter(csp, tracer)
   stats = solver.solve()  # API idéntica
   ```

### Escenario 2: Si el ArcEngine No Existe (Más Probable)

**Estrategia:** Implementar las optimizaciones directamente en CSPSolver

Este es el escenario que analicé en el documento anterior. La estrategia es:

1. **NO crear un "ArcEngine" separado**
2. **Extender CSPSolver con optimizaciones modulares**
3. **Mantener compatibilidad total**

```python
# lattice_weaver/core/csp_engine/optimizations.py

class OptimizedCSPSolver:
    """
    Wrapper que añade optimizaciones al CSPSolver sin modificarlo.
    Implementa las funcionalidades que se esperaban del "ArcEngine".
    """
    
    def __init__(self, base_solver: CSPSolver, 
                 use_ac31=True,           # AC-3.1 en lugar de AC-3
                 use_cache=True,          # Caché de revisiones
                 use_clustering=False,    # Clustering dinámico
                 use_adaptive=True):      # Selección adaptativa
        
        self.solver = base_solver
        self.use_ac31 = use_ac31
        self.use_cache = use_cache
        self.use_clustering = use_clustering
        self.use_adaptive = use_adaptive
        
        # Estructuras de optimización
        self._revision_cache = {} if use_cache else None
        self._last_support = {} if use_ac31 else None
        
    def enforce_arc_consistency(self) -> bool:
        """
        AC-3.1 optimizado (lo que debería hacer ArcEngine).
        """
        if self.use_ac31:
            return self._enforce_ac31()
        else:
            return self.solver.enforce_arc_consistency()
    
    def _enforce_ac31(self) -> bool:
        """
        AC-3.1 con last_support.
        Evita recomputar soportes que ya se encontraron.
        """
        queue = []
        
        # Inicializar cola y last_support
        for constraint in self.solver.csp.constraints:
            if len(constraint.scope) == 2:
                var_i, var_j = list(constraint.scope)
                queue.append((var_i, var_j, constraint))
                queue.append((var_j, var_i, constraint))
                
                # Inicializar last_support
                if self._last_support is not None:
                    for val in self.solver.csp.domains[var_i]:
                        self._last_support[(var_i, val, var_j)] = None
        
        while queue:
            var_i, var_j, constraint = queue.pop(0)
            
            if self._revise_ac31(var_i, var_j, constraint):
                if not self.solver.csp.domains[var_i]:
                    return False
                
                # Añadir vecinos
                for neighbor_constraint in self.solver.csp.constraints:
                    if len(neighbor_constraint.scope) == 2:
                        n_var1, n_var2 = list(neighbor_constraint.scope)
                        if n_var2 == var_i and n_var1 != var_j:
                            queue.append((n_var1, n_var2, neighbor_constraint))
                        elif n_var1 == var_i and n_var2 != var_j:
                            queue.append((n_var2, n_var1, neighbor_constraint))
        
        return True
    
    def _revise_ac31(self, var_i: str, var_j: str, constraint: Constraint) -> bool:
        """
        Revise con last_support (AC-3.1).
        """
        revised = False
        new_domain_i = []
        original_domain_i = list(self.solver.csp.domains[var_i])
        original_domain_j = list(self.solver.csp.domains[var_j])
        
        for x in original_domain_i:
            # Intentar usar last_support
            last_support_key = (var_i, x, var_j)
            last_support = self._last_support.get(last_support_key)
            
            # Verificar si last_support sigue siendo válido
            if last_support and last_support in original_domain_j:
                if self._check_support(x, last_support, var_i, var_j, constraint):
                    new_domain_i.append(x)
                    continue
            
            # Buscar nuevo soporte
            found_support = False
            for y in original_domain_j:
                if self._check_support(x, y, var_i, var_j, constraint):
                    new_domain_i.append(x)
                    self._last_support[last_support_key] = y  # Guardar soporte
                    found_support = True
                    break
            
            if not found_support:
                revised = True
        
        if revised:
            self.solver.csp.domains[var_i] = frozenset(new_domain_i)
        
        return revised
    
    def _check_support(self, val_i, val_j, var_i, var_j, constraint):
        """Verifica si val_j es soporte para val_i"""
        if var_i == list(constraint.scope)[0]:
            return constraint.relation(val_i, val_j)
        else:
            return constraint.relation(val_j, val_i)
    
    def solve(self, *args, **kwargs):
        """Delegar al solver base"""
        return self.solver.solve(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegar métodos no definidos al solver base"""
        return getattr(self.solver, name)
```

---

## 🎨 Recomendación Final

### Paso 1: Verificar Existencia del ArcEngine

**Preguntas para el equipo:**
1. ¿Existe una implementación completa del ArcEngine en algún lugar?
2. ¿En qué rama o commit se encuentra?
3. ¿Cuál es su API exacta?

### Paso 2A: Si ArcEngine Existe

**Plan:**
1. Identificar la rama/commit con ArcEngine
2. Hacer merge cuidadoso
3. Crear `ArcEngineAdapter` para compatibilidad con CSPSolver
4. Tests exhaustivos de compatibilidad
5. Migración gradual

**Tiempo estimado:** 20-30 horas  
**Riesgo:** Medio-Alto (merge complejo)

### Paso 2B: Si ArcEngine NO Existe (Recomendado)

**Plan:**
1. Implementar optimizaciones en `OptimizedCSPSolver`
2. Mantener CSPSolver actual sin cambios
3. Sistema de wrappers modulares
4. Tests de compatibilidad
5. Benchmarking

**Tiempo estimado:** 15-20 horas  
**Riesgo:** Bajo (no rompe nada)

---

## 📋 Checklist de Decisión

- [ ] **Verificar con el equipo:** ¿Dónde está el ArcEngine real?
- [ ] **Si existe:** ¿Vale la pena integrarlo o es legacy code?
- [ ] **Si no existe:** ¿Implementamos las optimizaciones desde cero?
- [ ] **Decisión de arquitectura:** ¿Adapter pattern o reimplementación?
- [ ] **Plan de tests:** ¿Cómo garantizamos que no rompemos nada?

---

## 🚨 Advertencia Importante

**NO implementar "a ciegas"** sin saber si el ArcEngine existe. Dos escenarios muy diferentes:

### Si existe:
- ✅ Aprovechar código existente
- ⚠️ Pero requiere merge complejo
- ⚠️ APIs incompatibles requieren adapter

### Si no existe:
- ✅ Libertad para diseñar óptimamente
- ✅ Integración más limpia
- ⚠️ Más trabajo de implementación

---

## 🎯 Próximos Pasos Inmediatos

1. **Contactar al equipo** para ubicar el ArcEngine real
2. **Si existe:** Revisar su código y API
3. **Si no existe:** Proceder con `OptimizedCSPSolver`
4. **En ambos casos:** Mantener CSPSolver actual intacto

**Conclusión:** Necesitamos **clarificar primero qué existe realmente** antes de decidir la estrategia de integración. La buena noticia es que en ambos casos podemos evitar romper el solver actual usando patrones de diseño apropiados.


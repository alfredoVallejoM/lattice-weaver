# Solvers CSP del Estado del Arte

## Solvers Modernos Principales

### 1. **Google OR-Tools CP-SAT**

**OR-Tools** es una suite de optimización de código abierto desarrollada por Google que incluye el solver **CP-SAT** (Constraint Programming - Satisfiability), considerado uno de los solvers CSP más avanzados y eficientes de la actualidad.

**Características principales:**
- Diseñado específicamente para problemas de programación entera y CSP.
- Utiliza técnicas avanzadas de propagación de restricciones y búsqueda.
- Integra técnicas de SAT solving con constraint programming.
- Soporta restricciones globales complejas.
- Excelente rendimiento en problemas de scheduling, asignación y optimización combinatoria.

**Rendimiento en N-Queens:**
- Resuelve el problema de 8-Queens encontrando las 92 soluciones de manera eficiente.
- Escalable a tableros grandes (N > 100) con técnicas de búsqueda optimizadas.
- Utiliza propagación de restricciones y backtracking inteligente.

**Uso:**
```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()
solver = cp_model.CpSolver()
status = solver.Solve(model)
```

**Referencias:**
- Documentación oficial: https://developers.google.com/optimization/cp/cp_solver
- Ejemplo N-Queens: https://developers.google.com/optimization/cp/queens
- GitHub: https://github.com/google/or-tools

---

### 2. **Gecode**

**Gecode** (Generic Constraint Development Environment) es un toolkit de código abierto para desarrollar sistemas de constraint programming basados en propagación.

**Características principales:**
- Implementación eficiente de algoritmos de propagación de restricciones.
- Soporta restricciones globales y restricciones definidas por el usuario.
- Altamente configurable y extensible.
- Excelente rendimiento en problemas de scheduling y asignación.

**Rendimiento:**
- Considerado uno de los solvers más rápidos para problemas de CSP clásicos.
- Excelente rendimiento en problemas de N-Queens, Sudoku y scheduling.
- Utiliza técnicas avanzadas de propagación y búsqueda.

**Referencias:**
- Sitio web oficial: https://www.gecode.org/
- Documentación: https://www.gecode.org/documentation.html

---

### 3. **Minion**

**Minion** es un solver CSP rápido y escalable desarrollado originalmente en 2006, diseñado para problemas de constraint satisfaction de gran escala.

**Características principales:**
- Diseñado para escalabilidad y rendimiento.
- Soporta restricciones globales y restricciones definidas por el usuario.
- Utiliza técnicas de propagación eficientes.
- Buen rendimiento en problemas de gran escala.

**Rendimiento:**
- Buen rendimiento en problemas de N-Queens y otros problemas clásicos.
- Puede tener dificultades con problemas muy grandes (N > 100 en N-Queens).
- Utiliza backtracking con propagación de restricciones.

**Referencias:**
- Sitio web: https://heather.cafe/software/minion/
- Paper original: "Minion: A fast scalable constraint solver"

---

### 4. **Choco Solver**

**Choco** es un solver CSP de código abierto escrito en Java, diseñado para ser fácil de usar y extensible.

**Características principales:**
- Implementación en Java, fácil de integrar en aplicaciones Java.
- Soporta restricciones globales y restricciones definidas por el usuario.
- Incluye técnicas de búsqueda avanzadas y heurísticas.
- Buen rendimiento en problemas de scheduling y optimización.

**Referencias:**
- Sitio web oficial: https://choco-solver.org/
- GitHub: https://github.com/chocoteam/choco-solver

---

### 5. **Chuffed**

**Chuffed** es un solver CSP con aprendizaje de conflictos (conflict learning), similar a las técnicas usadas en SAT solvers.

**Características principales:**
- Utiliza técnicas de aprendizaje de conflictos para mejorar la búsqueda.
- Buen rendimiento en problemas de optimización.
- Integrado con MiniZinc.

**Referencias:**
- GitHub: https://github.com/chuffed/chuffed

---

## Comparaciones de Rendimiento

### Benchmarks de XCSP3 Competition 2024

La **XCSP3 Competition 2024** es una competencia anual que evalúa el rendimiento de solvers CSP en una amplia variedad de problemas de benchmark.

**Resultados destacados:**
- **OR-Tools CP-SAT** consistentemente entre los mejores solvers en múltiples categorías.
- **Gecode** excelente rendimiento en problemas de scheduling y asignación.
- **Chuffed** buen rendimiento en problemas de optimización con aprendizaje de conflictos.

**Configuración de la competencia:**
- 150 instancias CSP y 155 instancias COP (Constraint Optimization Problems).
- Timeout de 1800 segundos por instancia.
- 64 GiB de RAM por solver.

**Referencias:**
- XCSP3 Competition 2024: https://cp2024.a4cp.org/slides/Others/resultsXCSP24.pdf
- XCSP3 Format: https://xcsp.org/

---

### Rendimiento en N-Queens

Según la literatura y benchmarks informales:

| Solver | N=8 | N=12 | N=20 | N=100 |
|--------|-----|------|------|-------|
| **OR-Tools CP-SAT** | < 0.01s | < 0.1s | < 1s | < 10s |
| **Gecode** | < 0.01s | < 0.1s | < 1s | < 10s |
| **Minion** | < 0.01s | < 0.1s | < 2s | Timeout |
| **Simple Backtracking** | < 0.01s | < 0.1s | ~10s | Timeout |

**Nota**: Los tiempos son aproximados y dependen del hardware y la implementación específica.

---

### Rendimiento en Sudoku

Según la literatura:

| Solver | Sudoku Fácil (9x9) | Sudoku Medio (9x9) | Sudoku Difícil (9x9) |
|--------|---------------------|---------------------|----------------------|
| **OR-Tools CP-SAT** | < 0.01s | < 0.1s | < 1s |
| **Gecode** | < 0.01s | < 0.1s | < 1s |
| **Minion** | < 0.01s | < 0.1s | < 2s |
| **Simple Backtracking** | < 0.1s | < 1s | Timeout |

**Nota**: Los tiempos son aproximados y dependen de la instancia específica y el hardware.

---

## Técnicas Avanzadas en Solvers Modernos

### 1. **Propagación de Restricciones Avanzada**

Los solvers modernos utilizan algoritmos de propagación de restricciones más sofisticados que el simple backtracking:

- **Arc Consistency (AC-3, AC-4)**: Eliminación de valores inconsistentes de los dominios.
- **Generalized Arc Consistency (GAC)**: Extensión de AC para restricciones n-arias.
- **Bound Consistency**: Propagación basada en límites de dominios.
- **Restricciones Globales**: Propagadores especializados para patrones comunes (AllDifferent, Global Cardinality, etc.).

### 2. **Heurísticas de Búsqueda**

- **Variable Ordering**: Selección inteligente de la siguiente variable a asignar (e.g., Minimum Remaining Values, Degree Heuristic).
- **Value Ordering**: Selección inteligente del valor a asignar (e.g., Least Constraining Value).
- **Restarts**: Reinicio de la búsqueda con diferentes configuraciones.
- **Large Neighborhood Search (LNS)**: Búsqueda local en vecindarios grandes.

### 3. **Aprendizaje de Conflictos**

Técnica inspirada en SAT solvers que permite al solver aprender de los conflictos encontrados durante la búsqueda:

- **Nogood Learning**: Almacenamiento de combinaciones de asignaciones que llevan a conflictos.
- **Clause Learning**: Aprendizaje de cláusulas que representan conflictos.
- **Backjumping**: Salto inteligente en el árbol de búsqueda basado en el análisis de conflictos.

### 4. **Paralelización**

Algunos solvers modernos utilizan paralelización para mejorar el rendimiento:

- **Portfolio Paralelo**: Ejecución de múltiples estrategias de búsqueda en paralelo.
- **Work Stealing**: Distribución dinámica del trabajo entre threads.
- **Shared Nogood Learning**: Compartición de nogoods aprendidos entre threads.

---

## Comparación con LatticeWeaver

### Rendimiento de LatticeWeaver vs Solvers del Estado del Arte

Basándonos en nuestros benchmarks, podemos hacer las siguientes observaciones:

#### **N-Queens 8x8**

| Solver | Tiempo Total (s) | Observaciones |
|--------|------------------|---------------|
| **LatticeWeaver (NoCompilation)** | ~0.0345 | Backtracking simple sin optimizaciones |
| **LatticeWeaver (L1-L6)** | ~0.0360-0.0396 | Compilación añade overhead sin beneficio |
| **OR-Tools CP-SAT** | < 0.01 | ~3-4x más rápido |
| **Gecode** | < 0.01 | ~3-4x más rápido |

**Conclusión**: El solver simple de LatticeWeaver es competitivo para problemas pequeños, pero los solvers del estado del arte son significativamente más rápidos debido a técnicas avanzadas de propagación y heurísticas.

#### **Sudoku 9x9**

| Solver | Tiempo Total (s) | Observaciones |
|--------|------------------|---------------|
| **LatticeWeaver** | Variable | Depende de la dificultad |
| **OR-Tools CP-SAT** | < 0.1 | Consistentemente rápido |
| **Gecode** | < 0.1 | Consistentemente rápido |

**Conclusión**: Para Sudoku, la diferencia es más pronunciada, especialmente en instancias difíciles, donde las técnicas avanzadas de propagación son cruciales.

#### **Graph Coloring**

| Solver | Tiempo Total (s) | Observaciones |
|--------|------------------|---------------|
| **LatticeWeaver** | Variable | Buen rendimiento en grafos dispersos |
| **OR-Tools CP-SAT** | < 0.1 | Excelente rendimiento en todos los casos |

**Conclusión**: El rendimiento de LatticeWeaver en Graph Coloring es variable, con buen rendimiento en grafos dispersos pero problemas en grafos densos.

---

## Oportunidades de Mejora para LatticeWeaver

### 1. **Implementar Propagación de Restricciones Avanzada**

El compilador multiescala de LatticeWeaver podría beneficiarse de técnicas de propagación más sofisticadas:

- **Arc Consistency (AC-3)**: Implementar AC-3 en los niveles inferiores del compilador para reducir dominios antes de la búsqueda.
- **Restricciones Globales**: Detectar patrones como AllDifferent en el compilador y usar propagadores especializados.

### 2. **Heurísticas de Búsqueda Inteligentes**

Actualmente, el solver simple usa backtracking básico. Implementar heurísticas de búsqueda podría mejorar significativamente el rendimiento:

- **Minimum Remaining Values (MRV)**: Seleccionar la variable con el dominio más pequeño.
- **Degree Heuristic**: Seleccionar la variable más restringida.
- **Least Constraining Value (LCV)**: Seleccionar el valor que menos restringe las variables futuras.

### 3. **Optimizar el Overhead de Compilación**

Los resultados muestran que el overhead de compilación no se compensa con mejoras en el tiempo de resolución. Esto sugiere que:

- El proceso de compilación es demasiado costoso para problemas pequeños.
- El compilador no está generando representaciones que faciliten la resolución.
- Se necesita un análisis de costo-beneficio para decidir cuándo aplicar la compilación.

### 4. **Integrar Aprendizaje de Conflictos**

Implementar técnicas de aprendizaje de conflictos podría mejorar el rendimiento en problemas difíciles:

- **Nogood Learning**: Almacenar combinaciones de asignaciones que llevan a conflictos.
- **Backjumping**: Saltar inteligentemente en el árbol de búsqueda.

---

## Referencias

1. **OR-Tools Documentation**: https://developers.google.com/optimization
2. **Gecode**: https://www.gecode.org/
3. **Minion**: https://heather.cafe/software/minion/
4. **CSPLib**: https://www.csplib.org/
5. **XCSP3 Competition**: https://www.xcsp.org/competitions/
6. **CP-SAT Primer**: https://github.com/d-krupke/cpsat-primer
7. **Handbook of Constraint Programming** (Rossi, van Beek, Walsh, 2006)
8. **Constraint Processing** (Dechter, 2003)


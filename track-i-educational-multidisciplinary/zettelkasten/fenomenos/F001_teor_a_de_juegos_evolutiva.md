---
id: F001
tipo: fenomeno
titulo: Teoría de Juegos Evolutiva
dominios: [economia, biologia, sociologia]
categorias: [C004]
tags: [juegos, equilibrio, evolucion, cooperacion, estrategia]
fecha_creacion: 2025-10-12
fecha_modificacion: 2025-10-12
estado: en_revision
prioridad: maxima
---

# Teoría de Juegos Evolutiva

## Descripción

La teoría de juegos evolutiva aplica conceptos de teoría de juegos clásica al estudio de la evolución de estrategias en poblaciones. A diferencia de la teoría de juegos tradicional que asume agentes racionales, la teoría evolutiva modela cómo las estrategias se propagan en una población basándose en su éxito reproductivo o fitness. Las estrategias más exitosas tienden a aumentar su frecuencia en la población a través de procesos de selección, mutación y deriva.

Este enfoque ha demostrado ser extraordinariamente fértil, con aplicaciones que van desde la biología evolutiva (explicando comportamientos cooperativos en animales) hasta la economía (modelando la evolución de normas sociales y estrategias de mercado) y la sociología (estudiando la emergencia de convenciones y cooperación social). El concepto central es la **Estrategia Evolutivamente Estable (ESS)**, una estrategia que, si es adoptada por la mayoría de la población, no puede ser invadida por ninguna estrategia alternativa rara.

## Componentes Clave

### Variables
- **Estrategia (s):** Regla de comportamiento que un individuo sigue en interacciones
- **Frecuencia de estrategia (x_i):** Proporción de la población que adopta la estrategia i
- **Payoff (π):** Beneficio obtenido por una estrategia en una interacción
- **Fitness (w):** Éxito reproductivo, típicamente función del payoff promedio

### Dominios
- **Estrategias:** Conjunto finito S = {s₁, s₂, ..., sₙ} de estrategias posibles
- **Frecuencias:** Simplex Δⁿ = {x ∈ ℝⁿ : xᵢ ≥ 0, Σxᵢ = 1}
- **Payoffs:** ℝ (números reales, pueden ser negativos)

### Restricciones/Relaciones
- **Conservación de población:** Σxᵢ = 1 (las frecuencias suman 1)
- **Matriz de payoffs:** A = [aᵢⱼ] donde aᵢⱼ es el payoff de estrategia i contra j
- **Payoff promedio:** π(sᵢ, x) = Σⱼ xⱼ aᵢⱼ (payoff de i contra población x)
- **Fitness promedio:** w̄(x) = Σᵢ xᵢ π(sᵢ, x)

### Función Objetivo
Encontrar **Estrategias Evolutivamente Estables (ESS)**: estrategias que no pueden ser invadidas por mutantes raros.

Formalmente, s* es ESS si para toda estrategia alternativa s ≠ s*, existe ε̄ > 0 tal que:
π(s*, εs + (1-ε)s*) > π(s, εs + (1-ε)s*) para todo 0 < ε < ε̄

## Mapeo a Formalismos

### CSP (Constraint Satisfaction Problem)

**Mapeo para encontrar equilibrios:**
- **Variables:** Frecuencias x = (x₁, ..., xₙ) de cada estrategia
- **Dominios:** xᵢ ∈ [0, 1] ∩ ℚ (racionales en [0,1] para discretización)
- **Restricciones:**
  - Σxᵢ = 1 (conservación)
  - Para equilibrio de Nash: xᵢ > 0 ⇒ π(sᵢ, x) = w̄(x)
  - Para ESS: condiciones de estabilidad adicionales
- **Tipo:** Optimización (maximizar fitness promedio) o Satisfacción (encontrar equilibrios)

**Observación:** El mapeo a CSP es más natural para juegos discretos con estrategias finitas. Para dinámicas continuas, ecuaciones diferenciales son más apropiadas.

### Sistemas Dinámicos (Replicator Dynamics)

**Formulación como sistema dinámico:**

ẋᵢ = xᵢ [π(sᵢ, x) - w̄(x)]

Donde:
- ẋᵢ: Tasa de cambio de frecuencia de estrategia i
- π(sᵢ, x): Payoff de estrategia i contra población x
- w̄(x): Fitness promedio de la población

**Puntos fijos:** Corresponden a equilibrios de Nash
**Puntos fijos asintóticamente estables:** Corresponden a ESS

### Teoría de Grafos

**Para juegos en redes:**
- **Nodos:** Individuos en la población
- **Aristas:** Interacciones posibles
- **Etiquetas de nodos:** Estrategia adoptada
- **Dinámica:** Actualización de estrategias basada en payoffs locales

## Ejemplos Concretos

### Ejemplo 1: Dilema del Prisionero Iterado

**Descripción:** Dos jugadores eligen cooperar (C) o desertar (D) repetidamente. Estrategias famosas incluyen Tit-for-Tat, Always Defect, Always Cooperate.

**Matriz de payoffs (por ronda):**
|       | C    | D    |
|-------|------|------|
| **C** | 3,3  | 0,5  |
| **D** | 5,0  | 1,1  |

**Parámetros:**
- Estrategias: {Always Cooperate, Always Defect, Tit-for-Tat, Pavlov, ...}
- Número de rondas: Variable o infinito con descuento
- Población: 100-1000 individuos

**Solución esperada:** En torneos de Axelrod (1984), Tit-for-Tat emergió como estrategia robusta. Es ESS bajo ciertas condiciones.

**Referencias:** 
- Axelrod, R. (1984). *The Evolution of Cooperation*. Basic Books.
- Nowak, M. A., & Sigmund, K. (1993). "A strategy of win-stay, lose-shift that outperforms tit-for-tat". *Nature*, 364, 56-58.

### Ejemplo 2: Halcón-Paloma (Hawk-Dove)

**Descripción:** Dos individuos compiten por un recurso de valor V. Halcones pelean (riesgo de costo C), Palomas se retiran.

**Matriz de payoffs:**
|       | H        | D      |
|-------|----------|--------|
| **H** | (V-C)/2  | V      |
| **D** | 0        | V/2    |

**Parámetros:**
- V = 10 (valor del recurso)
- C = 20 (costo de pelear)

**Solución esperada:** 
- Si V > C: ESS pura es Halcón
- Si V < C: ESS mixta con frecuencia de Halcón = V/C
- Para V=10, C=20: ESS es 50% Halcón, 50% Paloma

**Referencias:**
- Maynard Smith, J., & Price, G. R. (1973). "The logic of animal conflict". *Nature*, 246, 15-18.

### Ejemplo 3: Juegos de Coordinación - Driving Game

**Descripción:** Dos conductores se aproximan en direcciones opuestas. Deben coordinar para evitar colisión: ambos a la izquierda o ambos a la derecha.

**Matriz de payoffs:**
|       | L    | R    |
|-------|------|------|
| **L** | 1,1  | 0,0  |
| **R** | 0,0  | 1,1  |

**Solución esperada:** 
- Dos ESS puras: (L,L) y (R,R)
- Corresponden a convenciones sociales (conducir por la izquierda vs derecha)
- Históricamente, diferentes países convergieron a diferentes ESS

**Referencias:**
- Young, H. P. (1998). *Individual Strategy and Social Structure*. Princeton University Press.

## Conexiones

### Categoría Estructural
- [[C001]] - Redes de Interacción
- [[C004]] - Sistemas Dinámicos

### Conexiones Inversas
- [[C001]] - Redes de Interacción (instancia)
- [[C004]] - Sistemas Dinámicos (instancia)


### Isomorfismos
- [[I002]] - Dilema del Prisionero Multidominio - Aparece en biología, economía y sociología
- [[I001]] - Modelo de Ising ≅ Redes Sociales: Isomorfismo entre espines magnéticos y opiniones binarias.

### Instancias en Otros Dominios
- [[F003]] - Modelo de Ising 2D - Dinámicas de opinión análogas a espines magnéticos
- [[F002]] - Redes de Regulación Génica - Evolución de estrategias regulatorias

### Técnicas Aplicables
- Replicator Dynamics - Ecuaciones diferenciales que modelan evolución de frecuencias
- Simulación basada en agentes - Modelado computacional de poblaciones
- Análisis de estabilidad - Linearización alrededor de equilibrios
- Teoría de perturbaciones - Análisis de invasibilidad

### Conceptos Fundamentales
- Equilibrio de Nash - Concepto base de teoría de juegos
- Estrategia Evolutivamente Estable (ESS) - Extensión evolutiva del equilibrio
- Fitness - Medida de éxito reproductivo
- Invasibilidad - Capacidad de una estrategia mutante de invadir

## Propiedades Matemáticas

### Complejidad Computacional
- **Encontrar equilibrio de Nash (2 jugadores):** PPAD-completo
- **Encontrar ESS:** No se conoce caracterización completa de complejidad
- **Simulación de replicator dynamics:** Tiempo polinomial por paso, pero convergencia puede ser exponencial

### Propiedades Estructurales
- **Todo ESS es equilibrio de Nash**, pero no viceversa
- **En juegos simétricos 2x2:** Máximo 3 equilibrios (2 puros, 1 mixto)
- **Teorema de Folk:** En juegos repetidos infinitamente, cualquier payoff factible puede ser sostenido como equilibrio
- **Principio de exclusión competitiva:** En replicator dynamics, estrategias estrictamente dominadas se extinguen

### Teoremas Relevantes
- **Teorema de Maynard Smith (1973):** Condiciones suficientes para ESS
- **Teorema de Taylor-Jonker (1978):** Replicator dynamics preserva equilibrios de Nash
- **Teorema de Zeeman (1980):** Clasificación de dinámicas en juegos 2x2

## Visualización

### Tipos de Visualización Aplicables

1. **Simplex de frecuencias (para 3 estrategias):**
   - Triángulo donde cada vértice representa una estrategia pura
   - Puntos interiores representan mezclas
   - Flechas muestran dirección de replicator dynamics
   - Atractores (ESS) son sumideros

2. **Diagramas de fase:**
   - Ejes: Frecuencias de estrategias
   - Curvas: Trayectorias de la dinámica
   - Puntos fijos: Equilibrios
   - Regiones de atracción coloreadas

3. **Torneos evolutivos:**
   - Animación temporal de frecuencias de estrategias
   - Gráfico de líneas: Frecuencia vs tiempo
   - Permite ver invasiones y extinciones

4. **Matrices de payoffs interactivas:**
   - Heatmap de payoffs
   - Permite modificar valores y ver cómo cambian equilibrios

### Componentes Reutilizables
- Visualizador de sistemas dinámicos 2D/3D (compartido con [[C004]])
- Simulador de poblaciones (compartido con modelos epidemiológicos)
- Graficador de redes (para juegos en grafos)

## Recursos

### Literatura Clave

1. Maynard Smith, J. (1982). *Evolution and the Theory of Games*. Cambridge University Press.
   - Libro fundacional de la teoría de juegos evolutiva

2. Axelrod, R. (1984). *The Evolution of Cooperation*. Basic Books.
   - Torneos del Dilema del Prisionero, origen de Tit-for-Tat

3. Weibull, J. W. (1995). *Evolutionary Game Theory*. MIT Press.
   - Tratamiento matemático riguroso

4. Nowak, M. A. (2006). *Evolutionary Dynamics: Exploring the Equations of Life*. Harvard University Press.
   - Enfoque moderno con aplicaciones a biología

5. Hofbauer, J., & Sigmund, K. (1998). *Evolutionary Games and Population Dynamics*. Cambridge University Press.
   - Análisis matemático profundo de dinámicas

### Datasets

- **Axelrod Tournament Data:** Resultados de torneos históricos del Dilema del Prisionero
  - Disponible en: http://axelrod.readthedocs.io/
  
- **Animal Behavior Database:** Datos empíricos de comportamientos cooperativos en animales
  - Fuente: Dugatkin, L. A. (1997). *Cooperation Among Animals*

### Implementaciones Existentes

- **Axelrod Python Library:** https://github.com/Axelrod-Python/Axelrod
  - Simulación de torneos del Dilema del Prisionero con 200+ estrategias
  - Licencia: MIT

- **EGTtools:** https://github.com/Socrats/EGTtools
  - Herramientas para simulación y análisis de juegos evolutivos
  - Licencia: GPL-3.0

- **PyEvolGame:** https://github.com/marcharper/python-egt
  - Implementación de replicator dynamics y análisis de estabilidad
  - Licencia: MIT

### Código en LatticeWeaver
- **Módulo:** `lattice_weaver/phenomena/evolutionary_games/`
- **Tests:** `tests/phenomena/test_evolutionary_games.py`
- **Documentación:** `docs/phenomena/evolutionary_games.md`

## Estado de Implementación

### Fase 1: Investigación
- [x] Revisión bibliográfica completada
- [x] Ejemplos concretos identificados (Dilema Prisionero, Halcón-Paloma, Coordinación)
- [x] Datasets recopilados (Axelrod tournaments)
- [x] Documento de investigación creado

### Fase 2: Diseño
- [ ] Mapeo a CSP diseñado
- [ ] Arquitectura de código planificada
- [ ] Visualizaciones diseñadas

### Fase 3: Implementación
- [ ] Clases base implementadas
- [ ] Algoritmos implementados (replicator dynamics, ESS finder)
- [ ] Tests unitarios escritos
- [ ] Tests de integración escritos

### Fase 4: Visualización
- [ ] Componentes de visualización implementados
- [ ] Visualizaciones interactivas creadas (simplex, phase diagrams)
- [ ] Exportación de visualizaciones

### Fase 5: Documentación
- [ ] Documentación de API
- [ ] Tutorial paso a paso
- [ ] Ejemplos de uso (notebooks)
- [ ] Casos de estudio

### Fase 6: Validación
- [ ] Revisión por pares
- [ ] Validación con expertos del dominio
- [ ] Refinamiento basado en feedback

## Estimaciones

- **Tiempo de investigación:** 20 horas ✅
- **Tiempo de diseño:** 15 horas
- **Tiempo de implementación:** 40 horas
- **Tiempo de visualización:** 15 horas
- **Tiempo de documentación:** 10 horas
- **TOTAL:** 100 horas

## Notas Adicionales

### Ideas para Expansión

- Juegos espaciales (en grids o redes)
- Juegos con más de 2 jugadores
- Juegos asimétricos (roles diferentes)
- Coevolución de múltiples especies
- Juegos con estructura de población (demes, metapoblaciones)
- Aprendizaje y adaptación (no solo selección)

### Preguntas Abiertas

- ¿Cómo modelar racionalidad limitada en contexto evolutivo?
- ¿Cuál es la relación precisa entre ESS y atractores de replicator dynamics?
- ¿Cómo extender a juegos con información incompleta?

### Observaciones

La teoría de juegos evolutiva es un ejemplo paradigmático de **isomorfismo interdisciplinar**: la misma estructura matemática (replicator dynamics, ESS) aparece en biología (evolución de comportamientos), economía (evolución de estrategias de mercado), sociología (evolución de normas sociales) y ciencia política (evolución de instituciones). Este isomorfismo permite transferir técnicas de análisis y resultados entre dominios.

---

**Última actualización:** 2025-10-12  
**Responsable:** Agente Autónomo de Análisis


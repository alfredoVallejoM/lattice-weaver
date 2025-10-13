---
id: F003
tipo: fenomeno
titulo: Modelo de Ising 2D
dominios: [fisica_estadistica, ciencia_computacional, sociologia]
categorias: [C001, C004]
tags: [fisica, magnetismo, transiciones_fase, redes, monte_carlo]
fecha_creacion: 2025-10-12
fecha_modificacion: 2025-10-12
estado: en_revision
prioridad: maxima
---

# Modelo de Ising 2D

## Descripción

El modelo de Ising es uno de los modelos más estudiados en física estadística y mecánica estadística. Originalmente propuesto por Wilhelm Lenz en 1920 y resuelto analíticamente en 2D por Lars Onsager en 1944, el modelo describe el comportamiento de materiales ferromagnéticos mediante espines que interactúan con sus vecinos en una red. Cada espín puede estar en uno de dos estados (arriba ↑ o abajo ↓), y la energía del sistema depende de la alineación de espines vecinos.

El modelo de Ising 2D es extraordinariamente importante porque exhibe una **transición de fase** a temperatura crítica T_c: por debajo de T_c, los espines tienden a alinearse espontáneamente (fase ferromagnética ordenada); por encima de T_c, la agitación térmica destruye el orden (fase paramagnética desordenada). Esta transición es un fenómeno emergente que no puede entenderse estudiando espines individuales, sino que surge de las interacciones colectivas. El modelo ha sido aplicado mucho más allá de la física, modelando formación de opiniones, segregación social, propagación de información y dinámicas de redes neuronales.

## Componentes Clave

### Variables
- **Espín (s_i):** Estado del sitio i en la red, s_i ∈ {-1, +1} o {↓, ↑}
- **Configuración (σ):** Asignación de espines a todos los sitios, σ = (s₁, s₂, ..., s_N)
- **Magnetización (M):** Suma de todos los espines, M = Σᵢ sᵢ
- **Energía (E):** Función de la configuración que determina su probabilidad

### Dominios
- **Espines:** s_i ∈ {-1, +1}
- **Red:** Típicamente grid 2D cuadrado de tamaño L × L (N = L² sitios)
- **Temperatura:** T ∈ ℝ⁺ (en unidades de energía/k_B)
- **Campo externo:** h ∈ ℝ (campo magnético aplicado)

### Restricciones/Relaciones

**Hamiltoniano (función de energía):**

E(σ) = -J Σ_{⟨i,j⟩} sᵢ sⱼ - h Σᵢ sᵢ

Donde:
- J: Constante de acoplamiento (J > 0 favorece alineación, ferromagnético)
- ⟨i,j⟩: Pares de vecinos más cercanos
- h: Campo magnético externo

**Distribución de Boltzmann:**

P(σ) = (1/Z) exp(-βE(σ))

Donde:
- β = 1/(k_B T): Inverso de temperatura
- Z = Σ_σ exp(-βE(σ)): Función de partición (normalización)

### Función Objetivo

- **Encontrar estado fundamental:** Configuración que minimiza E(σ)
- **Calcular propiedades termodinámicas:** Magnetización promedio, susceptibilidad, calor específico
- **Simular dinámica:** Evolución temporal hacia equilibrio térmico
- **Detectar transición de fase:** Identificar temperatura crítica T_c

## Mapeo a Formalismos

### CSP (Constraint Satisfaction Problem)

**Mapeo para encontrar estado fundamental (T=0):**
- **Variables:** Espines s_i para cada sitio i
- **Dominios:** s_i ∈ {-1, +1}
- **Restricciones:** Minimizar energía E(σ) = -J Σ_{⟨i,j⟩} sᵢ sⱼ - h Σᵢ sᵢ
- **Tipo:** Optimización (encontrar configuración de mínima energía)

**Observación:** Este es un problema de **MAX-CUT** en el grafo de la red cuando h=0, que es NP-hard. Sin embargo, para redes planares (como grid 2D), existen algoritmos polinomiales.

### Teoría de Grafos

**Representación como grafo:**
- **Nodos:** Sitios de la red (N nodos)
- **Aristas:** Conexiones entre vecinos (cada nodo tiene 4 vecinos en grid 2D)
- **Pesos de aristas:** -J sᵢ sⱼ (contribución a la energía)
- **Problema:** MAX-CUT ponderado

### Mecánica Estadística

**Ensemble canónico:**
- **Espacio de fases:** 2^N configuraciones posibles
- **Función de partición:** Z(T) = Σ_σ exp(-E(σ)/k_B T)
- **Energía libre:** F = -k_B T ln Z
- **Observables:** ⟨M⟩, ⟨E⟩, susceptibilidad χ, calor específico C

**Solución exacta de Onsager (1944):**
Para h=0, en el límite termodinámico (N→∞):

T_c = 2J / (k_B ln(1 + √2)) ≈ 2.269 J/k_B

Magnetización espontánea para T < T_c:

M(T) = [1 - sinh⁻⁴(2J/k_B T)]^(1/8)

### Algoritmos de Monte Carlo

**Metropolis-Hastings:**
1. Partir de configuración inicial σ
2. Proponer cambio: Voltear espín aleatorio sᵢ → -sᵢ
3. Calcular ΔE = E(σ') - E(σ)
4. Aceptar con probabilidad min(1, exp(-βΔE))
5. Repetir hasta equilibrio

**Algoritmo de Wolff (cluster updates):**
- Más eficiente cerca de T_c
- Voltea clusters enteros de espines alineados
- Reduce critical slowing down

## Ejemplos Concretos

### Ejemplo 1: Transición de Fase en Grid 10×10

**Descripción:** Simulación de modelo de Ising en grid pequeño para visualizar transición de fase.

**Parámetros:**
- Tamaño: L = 10 (N = 100 sitios)
- J = 1.0 (unidades arbitrarias)
- h = 0 (sin campo externo)
- Temperaturas: T ∈ {1.0, 2.0, 2.269, 3.0, 4.0}

**Solución esperada:**
- T = 1.0 < T_c: Alta magnetización |M| ≈ 0.9, dominios grandes
- T = 2.269 ≈ T_c: Fluctuaciones críticas, dominios de todos los tamaños
- T = 4.0 > T_c: Baja magnetización |M| ≈ 0.1, configuración aleatoria

**Referencias:**
- Onsager, L. (1944). "Crystal statistics. I. A two-dimensional model with an order-disorder transition". *Physical Review*, 65(3-4), 117.

### Ejemplo 2: Modelo de Opiniones (Sociofísica)

**Descripción:** Individuos en una red social pueden tener opinión A (↑) o B (↓). Tienden a adoptar opinión de vecinos (J > 0) pero con ruido térmico (temperatura social).

**Parámetros:**
- Red: Grid 50×50 o red social real
- J = 1.0 (presión de conformidad)
- T variable (nivel de "ruido social" o independencia)
- h = 0.1 (sesgo mediático hacia opinión A)

**Solución esperada:**
- T bajo: Consenso emerge (todos A o todos B)
- T alto: Opiniones fragmentadas, sin consenso
- h > 0: Sesgo hacia opinión A

**Referencias:**
- Castellano, C., Fortunato, S., & Loreto, V. (2009). "Statistical physics of social dynamics". *Reviews of Modern Physics*, 81(2), 591.

### Ejemplo 3: Reconstrucción de Imágenes

**Descripción:** Imagen binaria ruidosa. Usar modelo de Ising para denoising: píxeles vecinos tienden a tener mismo valor.

**Parámetros:**
- Imagen: 100×100 píxeles, binaria (blanco/negro)
- Ruido: 20% de píxeles volteados aleatoriamente
- J = 2.0 (favorece suavidad)
- T = 1.5 (controla trade-off entre suavidad y fidelidad a datos)

**Solución esperada:**
- Minimizar energía E = -J Σ_{vecinos} sᵢsⱼ + λ Σᵢ (sᵢ - dᵢ)²
- Donde dᵢ es píxel observado (ruidoso)
- Resultado: Imagen suavizada que preserva bordes

**Referencias:**
- Geman, S., & Geman, D. (1984). "Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images". *IEEE TPAMI*, 6(6), 721-741.

## Conexiones

#- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
- [[F003]] - Conexión inversa con Fenómeno.
## Categoría Estructural
- [[C001]] - Redes de Interacción (grid 2D es un grafo)
- [[C004]] - Sistemas Dinámicos (evolución temporal hacia equilibrio)

### Conexiones Inversas
- [[C001]] - Redes de Interacción (instancia)
- [[C004]] - Sistemas Dinámicos (instancia)


#- [[F003]] - Conexión inversa con Fenómeno.
## Isomorfismos
- [[I001]] - Modelo de Ising ≅ Redes Sociales - Formación de opiniones
- **Redes neuronales de Hopfield:** Espines = neuronas, alineación = memoria
- **Percolación:** Transición de fase análoga
- **Segregación urbana (modelo de Schelling):** Preferencia por vecinos similares

### Instancias en Otros Dominios
- [[F001]] - Teoría de Juegos Evolutiva - Dinámicas de cooperación en redes
- [[F002]] - Redes de Regulación Génica - Activación/desactivación binaria
- **Modelo de votantes:** Dinámica de opiniones
- **Propagación de epidemias (SIS):** Susceptible/Infectado análogo a ↓/↑

### Técnicas Aplicables
- **Monte Carlo Metropolis:** Muestreo de configuraciones
- **Algoritmo de Wolff:** Cluster updates para eficiencia
- **Transfer matrix:** Solución exacta en 1D y 2D
- **Renormalization group:** Análisis de transiciones de fase
- **Mean field theory:** Aproximación para dimensiones altas

### Conceptos Fundamentales
- **Transición de fase:** Cambio cualitativo en comportamiento macroscópico
- **Temperatura crítica:** Punto de transición
- **Exponentes críticos:** Caracterizan comportamiento cerca de T_c
- **Universalidad:** Sistemas diferentes comparten mismos exponentes críticos
- **Ruptura espontánea de simetría:** Magnetización espontánea rompe simetría ↑↔↓

## Propiedades Matemáticas

### Complejidad Computacional

- **Encontrar estado fundamental (h=0):** NP-hard en general (MAX-CUT)
- **Para grafos planares (como grid 2D):** Polinomial (algoritmo de planar MAX-CUT)
- **Calcular función de partición Z:** #P-completo (contar configuraciones ponderadas)
- **Muestreo de configuraciones:** Polinomial por muestra (Monte Carlo)

**Observación:** Aunque calcular Z exactamente es intratable, muestrear configuraciones según distribución de Boltzmann es eficiente con Monte Carlo.

### Propiedades Estructurales

- **Simetría ↑↔↓:** Hamiltoniano invariante bajo voltear todos los espines (cuando h=0)
- **Ruptura espontánea de simetría:** Para T < T_c, estado fundamental rompe simetría
- **Ergodicidad:** Monte Carlo explora todo el espacio de configuraciones
- **Detailed balance:** Metropolis satisface balance detallado, garantiza convergencia a equilibrio

### Teoremas Relevantes

- **Teorema de Onsager (1944):** Solución exacta para T_c y magnetización en 2D
- **Teorema de Mermin-Wagner:** No hay orden de largo alcance en 1D a T > 0
- **Teorema de Peierls:** Existencia de transición de fase en 2D
- **Teorema de Lee-Yang:** Ceros de función de partición determinan transiciones de fase

## Visualización

### Tipos de Visualización Aplicables

1. **Heatmap de configuración:**
   - Grid 2D con espines coloreados (azul=↓, rojo=↑)
   - Animación temporal mostrando evolución
   - Permite ver formación de dominios y fluctuaciones

2. **Magnetización vs Temperatura:**
   - Gráfico de M(T) mostrando transición de fase
   - Curva característica con caída abrupta en T_c
   - Comparar con solución de Onsager

3. **Histograma de energía:**
   - Distribución de energías muestreadas
   - Muestra picos correspondientes a fases
   - Útil para detectar transiciones de primer orden

4. **Tamaño de clusters:**
   - Distribución de tamaños de dominios de espines alineados
   - En T_c: Ley de potencias (sin escala característica)
   - Lejos de T_c: Exponencial

5. **Correlación espacial:**
   - Función de correlación ⟨sᵢ sⱼ⟩ vs distancia |i-j|
   - Decaimiento exponencial para T > T_c
   - Decaimiento algebraico en T = T_c

### Componentes Reutilizables
- Visualizador de grids 2D (compartido con autómatas celulares)
- Gráficos de series temporales (compartido con [[C004]])
- Visualizador de distribuciones (histogramas, PDFs)

## Recursos

### Literatura Clave

1. Onsager, L. (1944). "Crystal statistics. I. A two-dimensional model with an order-disorder transition". *Physical Review*, 65(3-4), 117-149.
   - Solución exacta del modelo de Ising 2D

2. Baxter, R. J. (1982). *Exactly Solved Models in Statistical Mechanics*. Academic Press.
   - Tratado exhaustivo de modelos exactamente solubles

3. Newman, M. E. J., & Barkema, G. T. (1999). *Monte Carlo Methods in Statistical Physics*. Oxford University Press.
   - Métodos computacionales para simulación

4. Landau, D. P., & Binder, K. (2014). *A Guide to Monte Carlo Simulations in Statistical Physics* (4th ed.). Cambridge University Press.
   - Guía práctica de simulaciones

5. Yeomans, J. M. (1992). *Statistical Mechanics of Phase Transitions*. Oxford University Press.
   - Introducción accesible a transiciones de fase

### Datasets

- **Ising Model Simulations:** Configuraciones de equilibrio a diferentes temperaturas
  - Disponible en: https://github.com/topics/ising-model
  
- **Spin Glass Server:** Instancias de problemas de optimización relacionados
  - URL: https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/SPINGLASS/

### Implementaciones Existentes

- **scikit-ising:** https://github.com/tomchaplin/scikit-ising
  - Simulación de modelo de Ising con visualización
  - Licencia: MIT

- **Ising.jl (Julia):** https://github.com/johnmyleswhite/Ising.jl
  - Implementación eficiente en Julia
  - Licencia: MIT

- **IsingModel (Python):** https://github.com/rajeshrinet/ising-model
  - Implementación didáctica con animaciones
  - Licencia: MIT

- **ALPS (Algorithms and Libraries for Physics Simulations):** https://alps.comp-phys.org/
  - Suite completa para física estadística
  - Licencia: GPL

### Código en LatticeWeaver
- **Módulo:** `lattice_weaver/phenomena/ising_model/`
- **Tests:** `tests/phenomena/test_ising.py`
- **Documentación:** `docs/phenomena/ising_model.md`

## Estado de Implementación

### Fase 1: Investigación
- [x] Revisión bibliográfica completada
- [x] Ejemplos concretos identificados (transición de fase, opiniones, imágenes)
- [x] Datasets recopilados
- [x] Documento de investigación creado

### Fase 2: Diseño
- [ ] Mapeo a CSP diseñado
- [ ] Arquitectura de código planificada
- [ ] Visualizaciones diseñadas

### Fase 3: Implementación
- [ ] Clases base implementadas (IsingModel, Lattice, Spin)
- [ ] Algoritmos implementados (Metropolis, Wolff, exact ground state)
- [ ] Tests unitarios escritos
- [ ] Tests de integración escritos

### Fase 4: Visualización
- [ ] Componentes de visualización implementados
- [ ] Visualizaciones interactivas creadas
- [ ] Animaciones de evolución temporal
- [ ] Exportación de visualizaciones

### Fase 5: Documentación
- [ ] Documentación de API
- [ ] Tutorial paso a paso
- [ ] Ejemplos de uso (notebooks)
- [ ] Casos de estudio

### Fase 6: Validación
- [ ] Revisión por pares
- [ ] Validación con físicos estadísticos
- [ ] Comparación con resultados exactos de Onsager
- [ ] Refinamiento basado en feedback

## Estimaciones

- **Tiempo de investigación:** 15 horas ✅
- **Tiempo de diseño:** 10 horas
- **Tiempo de implementación:** 35 horas
- **Tiempo de visualización:** 12 horas
- **Tiempo de documentación:** 8 horas
- **TOTAL:** 80 horas

## Notas Adicionales

### Ideas para Expansión

- **Modelo de Ising 3D:** Transición de fase más realista para materiales reales
- **Modelo de Potts:** Generalización con q > 2 estados por espín
- **Spin glasses:** J aleatorio, frustración, múltiples mínimos locales
- **Modelo XY:** Espines continuos en el plano
- **Modelo de Heisenberg:** Espines vectoriales 3D
- **Ising en redes complejas:** Grafos no regulares, redes libres de escala
- **Dinámicas fuera de equilibrio:** Quenches, hysteresis

### Preguntas Abiertas

- ¿Cómo caracterizar rigurosamente transiciones de fase en sistemas finitos?
- ¿Cuál es la relación precisa entre exponentes críticos y dimensionalidad?
- ¿Cómo acelerar simulaciones Monte Carlo cerca de T_c (critical slowing down)?
- ¿Cómo aplicar modelo de Ising a redes sociales reales con topología compleja?

### Observaciones

El modelo de Ising es un **ejemplo paradigmático de universalidad**: sistemas físicos completamente diferentes (líquidos, magnetos, aleaciones binarias) exhiben el mismo comportamiento crítico cerca de transiciones de fase, caracterizado por los mismos exponentes críticos. Esta universalidad sugiere que los detalles microscópicos son irrelevantes para el comportamiento macroscópico, y solo la dimensionalidad y simetrías importan.

El **isomorfismo con formación de opiniones** es profundo: la "temperatura social" representa el nivel de ruido o independencia de pensamiento, J representa la presión de conformidad, y la transición de fase corresponde a la emergencia de consenso vs fragmentación. Este isomorfismo permite aplicar técnicas de física estadística a ciencias sociales, un campo conocido como **sociofísica**.

La conexión con **CSP** es también notable: encontrar el estado fundamental es un problema de optimización combinatoria (MAX-CUT), mientras que el muestreo de configuraciones a temperatura finita es un problema de muestreo probabilístico. Esto conecta física estadística con ciencia computacional y optimización.

---

**Última actualización:** 2025-10-12  
**Responsable:** Agente Autónomo de Análisis
- [[F004]]
- [[F008]]
- [[F009]]
- [[F010]]
- [[I003]]
- [[I004]]
- [[I005]]
- [[I006]]
- [[I008]]
- [[T003]]
- [[K005]]
- [[K007]]
- [[K010]]

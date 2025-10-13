---
id: F010
tipo: fenomeno
titulo: Segregación urbana (Schelling)
dominios: [sociologia, economia, urbanismo, modelos_basados_en_agentes]
categorias: [C001, C004]
tags: [modelos_basados_en_agentes, segregacion, autoorganizacion, sistemas_complejos, tipping_points]
fecha_creacion: 2025-10-12
fecha_modificacion: 2025-10-13
estado: completo  # borrador | en_revision | completo
prioridad: media  # maxima | alta | media | baja
---

# Segregación urbana (Schelling)

## Descripción

El **Modelo de Segregación de Schelling**, propuesto por el economista Thomas C. Schelling en 1971, es un modelo basado en agentes que demuestra cómo preferencias individuales moderadas por la similitud pueden conducir a patrones de segregación a gran escala, incluso cuando ningún individuo desea activamente la segregación. El modelo opera en un tablero (generalmente una cuadrícula) donde agentes de diferentes tipos (ej. dos etnias) se mueven si no están satisfechos con la composición de sus vecinos. La "satisfacción" se define por un umbral mínimo de vecinos similares. Si un agente tiene menos vecinos similares de lo que desea, se mueve a una celda vacía donde su satisfacción es mayor.

La principal revelación del modelo de Schelling es que la segregación no requiere de prejuicios extremos o intenciones maliciosas. Pequeñas preferencias por vivir cerca de individuos similares pueden amplificarse a través de interacciones locales y autoorganización, resultando en una segregación macroscópica y persistente. Este fenómeno es un ejemplo clásico de cómo las micro-motivaciones pueden llevar a macro-comportamientos inesperados y es fundamental para entender la dinámica de sistemas complejos en sociología, economía y urbanismo.

## Componentes Clave

### Variables
-   **Agentes (A_i):** Individuos de diferentes tipos (ej. Tipo X, Tipo O).
-   **Ubicación (x, y):** La posición de un agente en una cuadrícula (o espacio discreto).
-   **Tipo de Agente (T_i):** La categoría a la que pertenece el agente `i`.
-   **Vecindario (N_i):** El conjunto de celdas adyacentes a la ubicación del agente `i`.
-   **Umbral de Satisfacción (τ):** La fracción mínima de vecinos similares que un agente requiere para estar satisfecho.

### Dominios
-   **Dominio de Ubicación:** Celdas discretas en una cuadrícula 2D (ej. `(0,0)` a `(N-1, M-1)`).
-   **Dominio de Tipo de Agente:** {Tipo X, Tipo O} (o cualquier conjunto finito de tipos).
-   **Dominio de Umbral de Satisfacción:** [0, 1] (número real).

### Restricciones/Relaciones
-   **Regla de Movimiento:** Un agente `i` se mueve a una celda vacía si la fracción de vecinos de su mismo tipo en su vecindario actual es menor que su umbral de satisfacción `τ`.
-   **Vecindario de Moore/Von Neumann:** Define qué celdas son consideradas "vecinas" (ej. las 8 celdas circundantes).
-   **Celdas Vacías:** Los agentes solo pueden moverse a celdas que no estén ocupadas por otro agente.

### Función Objetivo (si aplica)
-   No hay una función objetivo global explícita que los agentes busquen optimizar. El sistema evoluciona hacia un estado de **equilibrio** donde todos los agentes están satisfechos con su ubicación, o hacia un estado de **segregación persistente**.
-   Se puede definir una métrica de segregación (ej. índice de disimilitud) para cuantificar el nivel de segregación emergente.

## Mapeo a Formalismos

### CSP (Constraint Satisfaction Problem)
-   **Variables:** La ubicación de cada agente en la cuadrícula.
-   **Dominios:** El conjunto de todas las celdas posibles en la cuadrícula.
-   **Restricciones:** Para cada agente, la restricción es que la fracción de vecinos similares debe ser mayor o igual a su umbral de satisfacción `τ`. Además, no puede haber dos agentes en la misma celda.
-   **Tipo:** Satisfacción (encontrar una configuración de agentes donde todos estén satisfechos).

### Sistemas Dinámicos (Basados en Agentes)
-   **Espacio de Estados:** El conjunto de todas las posibles configuraciones de agentes en la cuadrícula.
-   **Dinámica:** Las reglas de movimiento de los agentes definen una dinámica discreta en el espacio de estados. La dinámica es estocástica si los agentes se eligen aleatoriamente para moverse o si las celdas vacías se eligen aleatoriamente.
-   **Atractores:** Los estados de equilibrio (donde ningún agente insatisfecho desea moverse) son los atractores del sistema. Estos atractores pueden ser estados segregados o integrados, dependiendo de los parámetros.

## Ejemplos Concretos

### Ejemplo 1: Segregación Residencial en Ciudades
**Contexto:** Cómo diferentes grupos demográficos (ej. etnias, niveles socioeconómicos) se distribuyen en una ciudad, formando barrios homogéneos.

**Mapeo:**
-   Agentes = Familias o individuos de diferentes grupos.
-   Ubicación = Casas o parcelas en un mapa urbano.
-   Umbral de Satisfacción = Preferencia por tener un cierto porcentaje de vecinos del mismo grupo.

**Solución esperada:** Incluso con umbrales de satisfacción bajos (ej. solo el 30% de los vecinos deben ser similares), el modelo predice una segregación significativa, mostrando que la segregación no es necesariamente el resultado de un racismo extremo, sino de preferencias moderadas.

**Referencias:** Schelling, T. C. (1971). "Dynamic models of segregation". *Journal of Mathematical Sociology*, 1(2), 143-186.

### Ejemplo 2: Segregación de Especies en Ecosistemas
**Contexto:** Cómo diferentes especies de plantas o animales se distribuyen en un hábitat, formando parches homogéneos.

**Mapeo:**
-   Agentes = Individuos de diferentes especies.
-   Ubicación = Parcelas de terreno.
-   Umbral de Satisfacción = Preferencia por vivir cerca de individuos de la misma especie (ej. por recursos, protección).

**Solución esperada:** El modelo puede explicar la formación de patrones espaciales en ecología, donde especies similares tienden a agruparse, incluso si no hay una competencia directa por el espacio.

**Referencias:** D'Souza, R. M., & Nagler, J. (2013). "An ecological perspective on Schelling's segregation model". *Physical Review E*, 88(5), 052810.

### Ejemplo 3: Segregación de Opiniones en Redes Sociales
**Contexto:** Cómo individuos con opiniones similares tienden a agruparse en "cámaras de eco" o "burbujas de filtro" en plataformas en línea.

**Mapeo:**
-   Agentes = Usuarios de redes sociales.
-   Ubicación = Posición en un espacio de opinión (conceptual, no geográfico).
-   Umbral de Satisfacción = Preferencia por interactuar con usuarios que comparten opiniones similares.

**Solución esperada:** El modelo predice que incluso preferencias leves por la homofilia (interactuar con personas similares) pueden llevar a una polarización y segregación de opiniones a nivel de la red.

**Referencias:** Flache, A., Mäs, M., Feliciani, T., Chattoe-Brown, E., Deffuant, J., Huet, S., & Lorenz, J. (2017). "Models of social influence: Towards a unified framework". *Journal of Artificial Societies and Social Simulation*, 20(4), 2.

## Conexiones

#- [[F010]] - Conexión inversa con Fenómeno.
- [[F010]] - Conexión inversa con Fenómeno.
- [[F010]] - Conexión inversa con Fenómeno.
- [[F010]] - Conexión inversa con Fenómeno.
- [[F010]] - Conexión inversa con Fenómeno.
- [[F010]] - Conexión inversa con Fenómeno.
- [[F010]] - Conexión inversa con Fenómeno.
- [[F010]] - Conexión inversa con Fenómeno.
- [[F010]] - Conexión inversa con Fenómeno.
- [[F010]] - Conexión inversa con Fenómeno.
- [[F010]] - Conexión inversa con Fenómeno.
- [[F010]] - Conexión inversa con Fenómeno.
- [[F010]] - Conexión inversa con Fenómeno.
- [[F010]] - Conexión inversa con Fenómeno.
- [[F010]] - Conexión inversa con Fenómeno.
## Categoría Estructural
-   [[C001]] - Redes de Interacción: El modelo se basa en interacciones locales entre agentes en una red (implícita o explícita).
-   [[C004]] - Sistemas Dinámicos: La evolución de la configuración de agentes es un proceso dinámico que converge a estados de equilibrio.

### Conexiones Inversas
-   [[C001]] - Redes de Interacción (instancia)
-   [[C004]] - Sistemas Dinámicos (instancia)

#- [[F010]] - Conexión inversa con Fenómeno.
## Isomorfismos
-   [[I###]] - Modelo de Schelling ≅ Modelo de Ising (en el sentido de que ambos son modelos de interacción local que exhiben transiciones de fase y autoorganización).
-   [[I###]] - Modelo de Schelling ≅ Modelos de Autoorganización (ej. autómatas celulares, modelos de formación de patrones).

### Instancias en Otros Dominios
-   [[F009]] - Modelo de votantes: Ambos modelos exploran la dinámica de opiniones/estados en una población, aunque con reglas de interacción diferentes.
-   [[F003]] - Modelo de Ising 2D: Ambos son modelos de retículo con interacciones locales que generan patrones macroscópicos.

### Técnicas Aplicables
-   [[T###]] - Simulación Basada en Agentes (ABM): La técnica fundamental para implementar y estudiar el modelo de Schelling.
-   [[T###]] - Análisis de Sensibilidad: Para entender cómo el umbral de satisfacción `τ` y otros parámetros afectan el nivel de segregación.

### Conceptos Fundamentales
-   [[K###]] - Autoorganización
-   [[K###]] - Propiedades Emergentes
-   [[K###]] - Puntos de Inflexión (Tipping Points)
-   [[K###]] - Modelos Basados en Agentes

### Prerequisitos
-   [[K###]] - Conceptos Básicos de Probabilidad
-   [[K###]] - Conceptos de Sistemas Complejos

## Propiedades Matemáticas

### Complejidad Computacional
-   **Simulación:** La simulación del modelo de Schelling es computacionalmente eficiente para cuadrículas de tamaño moderado, permitiendo la exploración de un amplio espacio de parámetros.
-   **Análisis Analítico:** El análisis analítico es desafiante debido a la naturaleza estocástica y no lineal de las interacciones, aunque se han desarrollado aproximaciones de campo medio.

### Propiedades Estructurales
-   **Umbral Crítico:** Existe un umbral crítico para `τ` por debajo del cual el sistema permanece integrado, y por encima del cual la segregación emerge rápidamente.
-   **Robustez:** Los patrones de segregación pueden ser muy robustos y difíciles de revertir una vez establecidos.

### Teoremas Relevantes
-   Aunque no hay un "Teorema de Schelling" formal en el sentido de un teorema matemático, el modelo es una demostración computacional poderosa de cómo las micro-motivaciones pueden llevar a macro-fenómenos.

## Visualización

### Tipos de Visualización Aplicables
1.  **Cuadrícula Animada:** Mostrar la cuadrícula con los agentes de diferentes tipos coloreados, y animar sus movimientos a lo largo del tiempo, revelando la formación de clústeres segregados.
2.  **Métricas de Segregación:** Gráficos de índices de segregación (ej. índice de disimilitud, índice de exposición) en función del tiempo o del umbral `τ`.
3.  **Mapas de Calor:** Mostrar la densidad de diferentes tipos de agentes en la cuadrícula.

### Componentes Reutilizables
-   Componentes de visualización de cuadrículas 2D.
-   Animaciones de agentes moviéndose.
-   Generadores de números aleatorios.

## Recursos

### Literatura Clave
1.  Schelling, T. C. (1971). "Dynamic models of segregation". *Journal of Mathematical Sociology*, 1(2), 143-186.
2.  Schelling, T. C. (1978). *Micromotives and Macrobehavior*. W. W. Norton & Company.
3.  Epstein, J. M., & Axtell, R. (1996). *Growing Artificial Societies: Social Science from the Bottom Up*. Brookings Institution Press.

### Datasets
-   **Datos censales:** Datos de distribución demográfica en ciudades para comparar con los resultados del modelo.
-   **Cuadrículas sintéticas:** Para simular diferentes configuraciones iniciales y tamaños de población.

### Implementaciones Existentes
-   **NetLogo:** Plataforma popular para modelado basado en agentes, con una implementación clásica del modelo de Schelling.
-   **Python (Mesa, NumPy):** Implementaciones sencillas para simulación.

### Código en LatticeWeaver
-   **Módulo:** `lattice_weaver/phenomena/schelling_segregation/`
-   **Tests:** `tests/phenomena/test_schelling_segregation.py`
-   **Documentación:** `docs/phenomena/schelling_segregation.md`

## Estado de Implementación

### Fase 1: Investigación
-   [x] Revisión bibliográfica completada
-   [x] Ejemplos concretos identificados
-   [x] Datasets recopilados (referenciados)
-   [ ] Documento de investigación creado (integrado aquí)

### Fase 2: Diseño
-   [x] Mapeo a CSP diseñado
-   [x] Mapeo a otros formalismos (Sistemas Dinámicos Basados en Agentes)
-   [ ] Arquitectura de código planificada
-   [ ] Visualizaciones diseñadas

### Fase 3: Implementación
-   [ ] Clases base implementadas
-   [ ] Algoritmos implementados
-   [ ] Tests unitarios escritos
-   [ ] Tests de integración escritos

### Fase 4: Visualización
-   [ ] Componentes de visualización implementados
-   [ ] Visualizaciones interactivas creadas
-   [ ] Exportación de visualizaciones

### Fase 5: Documentación
-   [ ] Documentación de API
-   [ ] Tutorial paso a paso
-   [ ] Ejemplos de uso
-   [ ] Casos de estudio

### Fase 6: Validación
-   [ ] Revisión por pares
-   [ ] Validación con expertos del dominio
-   [ ] Refinamiento basado en feedback

## Estimaciones

-   **Tiempo de investigación:** 15 horas
-   **Tiempo de diseño:** 8 horas
-   **Tiempo de implementación:** 25 horas
-   **Tiempo de visualización:** 12 horas
-   **Tiempo de documentación:** 8 horas
-   **TOTAL:** 68 horas

## Notas Adicionales

### Ideas para Expansión
-   Explorar variantes del modelo con más de dos tipos de agentes, o con preferencias heterogéneas.
-   Estudiar el impacto de diferentes topologías de red (más allá de la cuadrícula) en la segregación.
-   Conexión con modelos de difusión de innovaciones o propagación de enfermedades.

### Preguntas Abiertas
-   ¿Cómo se pueden diseñar intervenciones para reducir la segregación una vez que se ha establecido?
-   ¿Qué papel juegan las preferencias por la diversidad en la mitigación de la segregación?

### Observaciones
-   El modelo de Schelling es un recordatorio poderoso de que las micro-motivaciones pueden llevar a macro-fenómenos inesperados, y que los sistemas complejos pueden generar patrones de segregación incluso sin intenciones explícitas de segregar.

---

**Última actualización:** 2025-10-13
**Responsable:** Agente Autónomo de Análisis
- [[C001]]
- [[C004]]
- [[I003]]
- [[I006]]
- [[T001]]
- [[T007]]
- [[K001]]
- [[K009]]
- [[K010]]
- [[F003]]
- [[F009]]

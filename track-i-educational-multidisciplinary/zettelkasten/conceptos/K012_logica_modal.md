---
id: K012
tipo: concepto
titulo: Lógica Modal
dominios: [logica, filosofia, inteligencia_artificial]
categorias: [C006] # Satisfacibilidad Lógica
tags: [modalidad, necesidad, posibilidad, mundos_posibles, epistemica, doxastica]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo
prioridad: alta
---

# Lógica Modal

## Descripción

La **lógica modal** es una de las lógicas no clásicas más estudiadas, que se ocupa del estudio deductivo de expresiones como "es necesario que" (□) y "es posible que" (◇) [21, 22]. Introduce operadores modales que permiten analizar la verdad de las proposiciones en diferentes "mundos posibles" o contextos. En argumentación, la lógica modal es crucial para formalizar argumentos sobre la posibilidad y la necesidad, analizar la fuerza de las inferencias y modelar creencias y conocimiento (lógicas epistémicas y doxásticas) [23].

## Componentes Clave

-   **Operadores Modales:** □ (necesidad) y ◇ (posibilidad).
-   **Mundos Posibles:** Contextos en los que las proposiciones pueden ser verdaderas o falsas.
-   **Relaciones de Accesibilidad:** Conexiones entre mundos posibles que definen diferentes sistemas modales (ej., K, T, S4, S5).

## Mapeo a Formalismos

### CSP

-   **Variables:** Proposiciones y "mundos posibles" o "instantes de tiempo".
-   **Dominios:** Los dominios de las proposiciones se expanden para incluir su valor de verdad en cada mundo/instante.
-   **Restricciones:** Las relaciones de accesibilidad entre mundos se modelan como restricciones que vinculan los valores de verdad de las proposiciones a través de estos contextos. Esto permite al `AdaptiveConsistencyEngine` de LatticeWeaver resolver problemas que involucren modalidades.

## Conexiones

### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)

### Conceptos Relacionados
- [[K006]] - Lógica Proposicional
- [[K007]] - Lógica de Predicados
- [[K013]] - Lógica Deóntica (a crear)
- [[K014]] - Lógica Temporal (a crear)

## Recursos

### Literatura Clave
1.  Garson, J. (2000). *Modal Logic*. Stanford Encyclopedia of Philosophy. Recuperado de https://plato.stanford.edu/entries/logic-modal/
2.  Grossi, D. (s.f.). *Doing Argumentation Theory in Modal Logic*. Recuperado de https://eprints.illc.uva.nl/id/document/862

## Estado de Implementación

- [x] Investigación completada
- [ ] Diseño de mapeo
- [ ] Implementación
- [ ] Visualización
- [ ] Documentación
- [ ] Tutorial

## Notas Adicionales

La lógica modal es fundamental para la metafísica, la epistemología y la ética, donde se discuten conceptos de necesidad, posibilidad, conocimiento y creencia.


### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)



### Conceptos Inversos
- [[K013]] - Lógica Deóntica
- [[K014]] - Lógica Temporal



### Conceptos Inversos
- [[K015]] - Lógica Difusa (Fuzzy Logic)



### Conceptos Inversos
- [[K015]] - Lógica Difusa (Fuzzy Logic)


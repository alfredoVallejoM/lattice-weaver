---
id: K002
tipo: concepto
titulo: Estrategia Evolutivamente Estable (ESS)
dominio_origen: biologia,economia,matematicas
categorias_aplicables: [C001, C004]
tags: [teoria_de_juegos_evolutiva, evolucion, estabilidad, seleccion_natural]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo  # borrador | en_revision | completo
---

# Concepto: Estrategia Evolutivamente Estable (ESS)

## Descripción

Una **Estrategia Evolutivamente Estable (ESS)** es un concepto central en la teoría de juegos evolutiva, que describe una estrategia que, una vez adoptada por una población, no puede ser invadida por ninguna estrategia mutante rara. En otras palabras, si la mayoría de los miembros de una población están utilizando una ESS, ningún individuo que adopte una estrategia diferente (mutante) puede obtener una mayor aptitud (fitness) en promedio. Es un estado de equilibrio dinámico en el que la selección natural no favorece ninguna otra estrategia.

## Origen

**Dominio de origen:** [[D001]] - Biología (Etología, Ecología), [[D005]] - Matemáticas
**Año de desarrollo:** 1973
**Desarrolladores:** John Maynard Smith y George R. Price.
**Contexto:** Desarrollado para aplicar la teoría de juegos al estudio de la evolución biológica, específicamente para entender por qué ciertos patrones de comportamiento animal son estables a lo largo del tiempo. El concepto de ESS permitió modelar la evolución de estrategias en poblaciones donde los individuos interactúan repetidamente y la aptitud de una estrategia depende de la frecuencia de otras estrategias en la población.

## Formulación

### Definición Formal

Una estrategia `I` es una ESS si, para cualquier estrategia alternativa `J` (donde `I ≠ J`):

1.  `E(I, I) > E(J, I)` (La estrategia `I` es una mejor respuesta contra sí misma que `J` contra `I`)

O bien, si `E(I, I) = E(J, I)` (ambas estrategias tienen el mismo pago contra `I`),

2.  `E(I, J) > E(J, J)` (La estrategia `I` es una mejor respuesta contra `J` que `J` contra `J`)

Donde `E(X, Y)` representa el pago esperado para un individuo que juega la estrategia `X` contra un individuo que juega la estrategia `Y`.

### Relación con el Equilibrio de Nash

Una ESS es un [[K001]] - Equilibrio de Nash en estrategias puras o mixtas, pero con una condición adicional de estabilidad frente a la invasión de mutantes. No todo Equilibrio de Nash es una ESS, ya que un Equilibrio de Nash puede ser vulnerable a la invasión si la estrategia mutante obtiene el mismo pago contra la estrategia dominante, pero un pago mayor contra sí misma.

## Análisis

### Propiedades

1.  **Estabilidad Evolutiva:** Una vez que una población adopta una ESS, es resistente a la invasión de estrategias alternativas raras.
2.  **Robustez:** Las ESS son robustas a pequeñas perturbaciones en la composición de la población.
3.  **No necesariamente óptimo social:** Al igual que el Equilibrio de Nash, una ESS no garantiza que la población alcance el resultado más beneficioso para el grupo en su conjunto (ej. el equilibrio de halcón-paloma).

### Limitaciones

1.  **Asunción de interacciones aleatorias:** A menudo asume que los individuos se encuentran y interactúan aleatoriamente en la población.
2.  **Estrategias puras o mixtas fijas:** El concepto original se centra en estrategias fijas, aunque se ha extendido a estrategias condicionales.
3.  **No considera la coevolución:** No modela explícitamente cómo múltiples estrategias podrían coevolucionar simultáneamente.

## Aplicabilidad

### Categorías Estructurales Aplicables

1.  [[C001]] - Redes de Interacción
    -   **Por qué funciona:** Permite analizar la estabilidad de estrategias de interacción (ej. cooperación vs. deserción) en redes sociales o ecológicas, donde la aptitud de un individuo depende de las estrategias de sus vecinos.
    -   **Limitaciones:** La estructura de la red puede influir en la emergencia y estabilidad de las ESS, lo que añade complejidad al análisis.

2.  [[C004]] - Sistemas Dinámicos
    -   **Por qué funciona:** Las ESS representan puntos fijos o atractores en la dinámica de poblaciones que evolucionan sus estrategias. La [[T001]] - Replicator Dynamics es una herramienta clave para estudiar cómo las poblaciones convergen hacia una ESS.
    -   **Limitaciones:** La convergencia a una ESS no siempre está garantizada, y la dinámica puede ser cíclica o caótica en algunos juegos.

### Fenómenos Donde Se Ha Aplicado

-   [[F001]] - Teoría de Juegos Evolutiva: El concepto fundacional para entender la evolución de comportamientos como el altruismo, la agresión, la cooperación y la señalización en animales y humanos.
-   [[F009]] - Modelo de votantes: Puede usarse para analizar la estabilidad de las opiniones o preferencias políticas en una población.

## Conexiones
#- [[K002]] - Conexión inversa con Concepto.
- [[K002]] - Conexión inversa con Concepto.
- [[K002]] - Conexión inversa con Concepto.
- [[K002]] - Conexión inversa con Concepto.
- [[K002]] - Conexión inversa con Concepto.
- [[D001]] - Conexión inversa con Dominio.
- [[D005]] - Conexión inversa con Dominio.

## Técnicas Relacionadas

-   [[T001]] - Replicator Dynamics: Una ecuación diferencial que describe cómo las frecuencias de las estrategias en una población cambian con el tiempo, a menudo convergiendo hacia una ESS.
-   [[T005]] - Algoritmo Genético: Puede simular la evolución de estrategias en una población, donde las estrategias que son ESS tenderán a dominar.

#- [[K002]] - Conexión inversa con Concepto.
## Conceptos Fundamentales Relacionados

-   [[K001]] - Equilibrio de Nash: La ESS es un refinamiento del Equilibrio de Nash, añadiendo la condición de estabilidad evolutiva.
-   [[K009]] - Autoorganización: La emergencia de una ESS en una población es un ejemplo de autoorganización, donde un patrón estable surge de interacciones individuales.
-   [[K010]] - Emergencia: La estabilidad de una estrategia a nivel poblacional es una propiedad emergente de las interacciones y la selección a nivel individual.

## Historia y Evolución

### Desarrollo Histórico

-   **1973:** John Maynard Smith y George R. Price introducen el concepto de ESS en su paper "The Logic of Animal Conflict".
-   **1982:** John Maynard Smith publica "Evolution and the Theory of Games", consolidando el campo de la teoría de juegos evolutiva.
-   **Décadas siguientes:** Extensión del concepto a estrategias condicionales, juegos repetidos y aplicaciones en economía y ciencias sociales.

### Impacto

El concepto de ESS ha sido fundamental para la biología evolutiva, proporcionando un marco matemático riguroso para entender la evolución de los comportamientos sociales. Ha permitido explicar fenómenos como la agresión ritualizada, el altruismo recíproco y la proporción de sexos en la naturaleza. Su influencia se ha extendido a la economía, la antropología y la ciencia política, donde se utiliza para analizar la estabilidad de normas y convenciones sociales.

**Citaciones:** El trabajo de Maynard Smith y Price es altamente citado en biología evolutiva y teoría de juegos.
**Adopción:** Ampliamente adoptado en etología, ecología, sociobiología, economía evolutiva y ciencias sociales computacionales.

---

**Última actualización:** 2025-10-13  
**Responsable:** Manus AI  
**Validado por:** [Pendiente de revisión por experto humano]
- [[F001]]
- [[I006]]
- [[K001]]
- [[K004]]
- [[K005]]
- [[K009]]
- [[K010]]
- [[C001]]
- [[C004]]
- [[F009]]
- [[T001]]
- [[T005]]

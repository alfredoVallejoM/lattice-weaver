---
id: K001
tipo: concepto
titulo: Equilibrio de Nash
dominio_origen: economia,matematicas,ciencias_sociales
categorias_aplicables: [C001, C004]
tags: [teoria_de_juegos, equilibrio, estrategia, interaccion_racional]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo  # borrador | en_revision | completo
---

# Concepto: Equilibrio de Nash

## Descripción

El **Equilibrio de Nash** es un concepto de solución en la teoría de juegos que describe una situación en la que cada jugador, conociendo las estrategias de los demás, no tiene ningún incentivo para cambiar unilateralmente su propia estrategia. En otras palabras, es un conjunto de estrategias, una para cada jugador, tal que ningún jugador puede mejorar su resultado cambiando su estrategia mientras los demás mantengan las suyas. Es un estado de estabilidad en el que las decisiones de todos los agentes son óptimas dadas las decisiones de los demás.

## Origen

**Dominio de origen:** [[D007]] - Economía, [[D005]] - Matemáticas
**Año de desarrollo:** 1950
**Desarrolladores:** John Forbes Nash Jr.
**Contexto:** John Nash introdujo este concepto en su tesis doctoral de 1950, "Non-Cooperative Games", generalizando el concepto de equilibrio de Cournot para juegos no cooperativos. Su trabajo revolucionó la teoría económica y la forma en que se analizan las interacciones estratégicas en una amplia gama de disciplinas.

## Formulación

### Definición Formal

Un conjunto de estrategias `(s_1*, s_2*, ..., s_n*)` es un Equilibrio de Nash si para cada jugador `i`, la estrategia `s_i*` es una mejor respuesta a las estrategias de los demás jugadores `(s_1*, ..., s_{i-1}*, s_{i+1}*, ..., s_n*)`. Formalmente, para cada jugador `i` y para cualquier estrategia alternativa `s_i` de `i`:

`U_i(s_1*, ..., s_i*, ..., s_n*) ≥ U_i(s_1*, ..., s_i, ..., s_n*)`

Donde `U_i` es la función de utilidad (o pago) del jugador `i`.

### Tipos de Equilibrio de Nash

1.  **Equilibrio de Nash en Estrategias Puras:** Cada jugador elige una estrategia específica con certeza.
2.  **Equilibrio de Nash en Estrategias Mixtas:** Cada jugador elige una distribución de probabilidad sobre sus estrategias puras, es decir, elige aleatoriamente entre sus estrategias puras con ciertas probabilidades.

### Teorema de Existencia

**Teorema de Nash (1950):** Todo juego finito con un número finito de jugadores y un número finito de estrategias puras para cada jugador tiene al menos un Equilibrio de Nash en estrategias mixtas.

## Análisis

### Propiedades

1.  **Estabilidad:** Una vez que un sistema alcanza un Equilibrio de Nash, ningún jugador tiene un incentivo unilateral para desviarse.
2.  **No necesariamente óptimo de Pareto:** Un Equilibrio de Nash puede no ser la mejor solución posible para todos los jugadores colectivamente (ej. [[F001]] - Dilema del Prisionero).
3.  **Puede no ser único:** Un juego puede tener múltiples Equilibrios de Nash.

### Limitaciones

1.  **Asunción de racionalidad perfecta:** Asume que todos los jugadores son perfectamente racionales y buscan maximizar su propia utilidad.
2.  **Información completa:** Asume que los jugadores tienen información completa sobre el juego, incluyendo las funciones de utilidad de los demás.
3.  **Problemas de coordinación:** Si hay múltiples Equilibrios de Nash, los jugadores pueden tener dificultades para coordinarse en cuál elegir.

## Aplicabilidad

### Categorías Estructurales Aplicables

1.  [[C001]] - Redes de Interacción
    -   **Por qué funciona:** El Equilibrio de Nash es fundamental para analizar las interacciones estratégicas en redes, donde las decisiones de un agente afectan a sus vecinos y viceversa. Permite entender la estabilidad de los comportamientos en sistemas interconectados.
    -   **Limitaciones:** La complejidad computacional para encontrar Equilibrios de Nash aumenta exponencialmente con el número de jugadores y estrategias en redes grandes.

2.  [[C004]] - Sistemas Dinámicos
    -   **Por qué funciona:** En la teoría de juegos evolutiva, los Equilibrios de Nash (o sus análogos, como las Estrategias Evolutivamente Estables) representan puntos fijos o atractores en la dinámica de poblaciones que aprenden o evolucionan sus estrategias.
    -   **Limitaciones:** La dinámica de ajuste hacia un Equilibrio de Nash puede ser compleja y no siempre converge, o puede converger a un equilibrio subóptimo.

### Fenómenos Donde Se Ha Aplicado

-   [[F001]] - Teoría de Juegos Evolutiva: El concepto de Equilibrio de Nash es central para entender la estabilidad de las estrategias en poblaciones que evolucionan.
-   [[F009]] - Modelo de votantes: Puede analizarse la estabilidad de las opiniones como un tipo de equilibrio en la interacción social.
-   [[F010]] - Segregación urbana (Schelling): Aunque no es un juego formal, el modelo de Schelling muestra un equilibrio donde ningún agente tiene incentivo para moverse dadas las posiciones de los demás.

## Conexiones
#- [[K001]] - Conexión inversa con Concepto.
- [[K001]] - Conexión inversa con Concepto.
- [[K001]] - Conexión inversa con Concepto.
- [[K001]] - Conexión inversa con Concepto.
- [[K001]] - Conexión inversa con Concepto.
- [[D005]] - Conexión inversa con Dominio.
- [[D007]] - Conexión inversa con Dominio.

## Técnicas Relacionadas

-   [[T001]] - Replicator Dynamics: Una técnica utilizada en teoría de juegos evolutiva para modelar cómo las proporciones de diferentes estrategias en una población cambian con el tiempo, convergiendo a menudo hacia un Equilibrio de Nash.

#- [[K001]] - Conexión inversa con Concepto.
## Conceptos Fundamentales Relacionados

-   [[K002]] - Estrategia Evolutivamente Estable (ESS): Un concepto más fuerte que el Equilibrio de Nash, específico para la teoría de juegos evolutiva, donde una estrategia no solo es una mejor respuesta, sino que también es resistente a la invasión de estrategias mutantes raras.
-   [[K005]] - Atractores: En sistemas dinámicos, un Equilibrio de Nash puede ser visto como un atractor hacia el cual el sistema tiende a evolucionar.

## Historia y Evolución

### Desarrollo Histórico

-   **1928:** John von Neumann publica el teorema minimax para juegos de suma cero.
-   **1944:** Von Neumann y Oskar Morgenstern publican "Theory of Games and Economic Behavior", sentando las bases de la teoría de juegos.
-   **1950:** John Nash introduce el concepto de equilibrio para juegos no cooperativos.
-   **1994:** John Nash recibe el Premio Nobel de Economía por su trabajo.

### Impacto

El Equilibrio de Nash es uno de los conceptos más influyentes en la economía y las ciencias sociales. Ha proporcionado un marco para analizar una vasta gama de interacciones estratégicas, desde la competencia entre empresas hasta la política internacional y la biología evolutiva. Su universalidad y aplicabilidad a problemas complejos lo han convertido en una herramienta indispensable para entender el comportamiento racional en sistemas interactivos.

**Citaciones:** El trabajo de Nash es uno de los más citados en economía y teoría de juegos.
**Adopción:** Ampliamente adoptado en economía, ciencias políticas, sociología, biología, informática y psicología.

---

**Última actualización:** 2025-10-13  
**Responsable:** Manus AI  
**Validado por:** [Pendiente de revisión por experto humano]
- [[F001]]
- [[I006]]
- [[K002]]
- [[K004]]
- [[K005]]
- [[K010]]
- [[C001]]
- [[C004]]
- [[F009]]
- [[F010]]
- [[T001]]

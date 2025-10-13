---
id: F015
tipo: fenomeno
titulo: Marcos de Argumentación Basados en Supuestos (ABAs)
dominios: [logica, inteligencia_artificial, filosofia]
categorias: [C006, C007] # Satisfacibilidad Lógica, Sistemas de Razonamiento
tags: [argumentacion, supuestos, reglas, inferencia, no_monotonica]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo
prioridad: media
---

# Marcos de Argumentación Basados en Supuestos (ABAs)

## Descripción

Los **Marcos de Argumentación Basados en Supuestos** (Assumption-based Argumentation Frameworks, ABAs) definen los argumentos como conjuntos de reglas y los ataques en términos de supuestos y sus contrarios. Un ABA es una tupla `(L, R, A, -)`, donde `L` es el lenguaje, `R` es un conjunto de reglas de inferencia, `A` es un conjunto de supuestos, y `-` es una función que mapea cada supuesto a su contrario [16]. Un argumento en ABA es una prueba de una afirmación a partir de un conjunto de supuestos. Los ataques se definen cuando el contrario de un supuesto en un argumento puede ser probado por otro argumento [16]. Son particularmente útiles en el razonamiento no monotónico y en la representación de conocimiento con excepciones.

## Componentes Clave

-   **Lenguaje (L):** Conjunto de fórmulas lógicas.
-   **Reglas de Inferencia (R):** Permiten derivar conclusiones a partir de premisas.
-   **Supuestos (A):** Conjunto de proposiciones que pueden ser asumidas.
-   **Función de Contrario (-):** Mapea cada supuesto a su contrario.

## Mapeo a Formalismos

### CSP

Los ABAs se pueden modelar con CSPs:

-   **Variables:** Los supuestos `a ∈ A` y las afirmaciones `c ∈ L` pueden ser variables con dominios de verdad (Verdadero/Falso).
-   **Restricciones:** Las reglas de inferencia `R` y la función `contrario` (`-`) se traducen en restricciones. Si una regla `s0 ← s1, ..., sm` existe, entonces `s0` es verdadero si `s1, ..., sm` son verdaderos. Si `a` es un supuesto y `a` es verdadero, entonces su contrario `a-` debe ser falso. Los ataques se modelan como restricciones que impiden que un supuesto y su contrario sean ambos verdaderos.

## Ejemplos Concretos

1.  **Argumento por Defecto:** "Los pájaros vuelan (regla). Tweety es un pájaro (supuesto). Por lo tanto, Tweety vuela." Si aparece la información "Tweety es un pingüino" (contrario del supuesto), el argumento es atacado.

## Conexiones

### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)

### Técnicas Aplicables
- [[T001]] - Constraint Propagation

### Conceptos Relacionados
- [[K006]] - Lógica Proposicional
- [[K007]] - Lógica de Predicados
- [[K010]] - Semánticas de Aceptación (Argumentación)

## Recursos

### Literatura Clave
1.  Dung, P. M., Kowalski, R. A., & Toni, F. (2009). *Assumption-based argumentation*. In Argumentation in Artificial Intelligence (pp. 199-218). Springer, Berlin, Heidelberg.

## Estado de Implementación

- [x] Investigación completada
- [ ] Diseño de mapeo
- [ ] Implementación
- [ ] Visualización
- [ ] Documentación
- [ ] Tutorial

## Notas Adicionales

Los ABAs proporcionan un marco flexible para el razonamiento no monotónico, permitiendo la revisión de creencias ante nueva información.


### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)


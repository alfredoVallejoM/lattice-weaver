---
id: F014
tipo: fenomeno
titulo: Marcos de Argumentación Basados en Valores (VAFs)
dominios: [logica, inteligencia_artificial, filosofia, etica]
categorias: [C007] # Sistemas de Razonamiento
tags: [argumentacion, valores, preferencias, etica, decision]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo
prioridad: media
---

# Marcos de Argumentación Basados en Valores (VAFs)

## Descripción

Los **Marcos de Argumentación Basados en Valores** (Value-based Argumentation Frameworks, VAFs) extienden los Marcos de Argumentación Abstractos (AFs) al incorporar valores que los argumentos promueven. Un VAF se define como una tupla `(A, R, V, val, valprefs)`, donde `A` es un conjunto de argumentos, `R` es una relación de ataque, `V` es un conjunto de valores, `val` mapea argumentos a valores, y `valprefs` es una relación de preferencia entre valores [16]. En un VAF, un ataque de `a` a `b` solo tiene éxito si el valor promovido por `b` no es preferido al valor promovido por `a`. Esto permite modelar situaciones donde la fuerza de un argumento depende de los valores subyacentes que defiende [16]. Son particularmente relevantes en la argumentación ética y legal.

## Componentes Clave

-   **Argumentos (A):** Entidades abstractas.
-   **Relación de Ataque (R):** Relación binaria entre argumentos.
-   **Valores (V):** Conjunto de principios o bienes que los argumentos promueven.
-   **Función `val`:** Asigna un valor a cada argumento.
-   **Preferencias de Valores (`valprefs`):** Relación de orden entre los valores.

## Mapeo a Formalismos

### CSP

Los VAFs pueden extender el mapeo a CSPs:

-   **Variables:** Además de los argumentos, los valores `v ∈ V` también pueden ser variables, con dominios que representen su prioridad o aceptabilidad.
-   **Restricciones:** Las preferencias `valprefs` se traducen en restricciones que afectan la resolución de conflictos. Si `a` ataca `b`, la restricción de ataque solo se activa si `val(b)` no es preferido a `val(a)`. Esto introduce una capa de meta-restricciones que el `AdaptiveConsistencyEngine` de LatticeWeaver podría manejar.

### FCA

-   **Objetos:** Argumentos o conjuntos de argumentos.
-   **Atributos:** Los valores asociados a los argumentos (`val(a)`) y las preferencias entre valores (`valprefs`) pueden ser atributos, permitiendo analizar cómo los valores estructuran el espacio de argumentos.

## Ejemplos Concretos

1.  **Dilema Ético:** Un argumento `A` promueve la "libertad" y ataca `B` que promueve la "seguridad". Si la "libertad" es preferida a la "seguridad", el ataque de `A` a `B` es exitoso.

## Conexiones

### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)
- [[F012]] - Marcos de Argumentación Abstractos (Dung's Framework)

### Técnicas Aplicables
- [[T001]] - Constraint Propagation

### Conceptos Relacionados
- [[K010]] - Semánticas de Aceptación (Argumentación)
- [[K011]] - Valores y Preferencias (a crear)

## Recursos

### Literatura Clave
1.  Bench-Capon, T. (2002). *Value-based argumentation frameworks*. 9th International Workshop on Non-Monotonic Reasoning (NMR 2002): 443–454.

## Estado de Implementación

- [x] Investigación completada
- [ ] Diseño de mapeo
- [ ] Implementación
- [ ] Visualización
- [ ] Documentación
- [ ] Tutorial

## Notas Adicionales

Los VAFs son esenciales para modelar la argumentación en dominios donde los principios morales o éticos juegan un papel crucial.


### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)



### Conceptos Inversos
- [[F012]] - Marcos de Argumentación Abstractos (Dung's Framework)



### Conceptos Inversos
- [[K011]] - Valores y Preferencias


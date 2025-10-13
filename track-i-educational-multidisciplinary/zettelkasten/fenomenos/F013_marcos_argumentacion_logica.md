---
id: F013
tipo: fenomeno
titulo: Marcos de Argumentación Basados en Lógica
dominios: [logica, inteligencia_artificial, filosofia]
categorias: [C006, C007] # Satisfacibilidad Lógica, Sistemas de Razonamiento
tags: [argumentacion, logica, inferencia, consistencia, conocimiento]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo
prioridad: media
---

# Marcos de Argumentación Basados en Lógica

## Descripción

En contraste con los Marcos de Argumentación Abstractos (AFs), los **Marcos de Argumentación Basados en Lógica** (Logic-based Argumentation Frameworks) otorgan una estructura interna a los argumentos. Un argumento se define como un par `(Φ, α)`, donde `Φ` es un conjunto mínimo y consistente de fórmulas lógicas que prueba la conclusión `α` [16]. La relación de ataque no se da explícitamente, sino que se deriva de propiedades lógicas. Por ejemplo, un argumento `(Ψ, β)` puede atacar a `(Φ, α)` si `β` contradice alguna de las premisas de `Φ` (defeater), o si `β` es la negación de `α` (rebuttal) [16]. Estos marcos son cruciales para modelar el razonamiento donde la validez de los argumentos depende de su contenido lógico.

## Componentes Clave

-   **Argumento:** Un par `(Φ, α)`, donde `Φ` es un conjunto de premisas lógicas y `α` es la conclusión derivada.
-   **Estructura Interna:** Los argumentos no son atómicos, sino que tienen una composición lógica.
-   **Relación de Ataque:** Derivada de propiedades lógicas, como contradicción de premisas o negación de conclusiones.

## Mapeo a Formalismos

### CSP

-   **Variables:** Las proposiciones atómicas o fórmulas lógicas dentro de los argumentos `(Φ, α)` pueden ser variables. Sus dominios serían valores de verdad (Verdadero/Falso).
-   **Restricciones:** Las reglas de inferencia y las relaciones de ataque (defeater, undercut, rebuttal) se convierten en restricciones lógicas que vinculan los valores de verdad de las proposiciones. Por ejemplo, si `Φ` prueba `α`, esto impone una restricción sobre los valores de verdad de las fórmulas en `Φ` y `α`.

### FCA

-   **Objetos:** Los argumentos `(Φ, α)` mismos pueden ser objetos en un contexto formal.
-   **Atributos:** Las propiedades lógicas de `Φ` y `α` (consistencia, validez, tipo de inferencia), así como las relaciones de ataque específicas, pueden ser atributos. Esto permitiría construir un lattice conceptual que clasifique los argumentos según sus propiedades lógicas y sus interacciones.

## Ejemplos Concretos

1.  **Argumento por Contradicción:** Un argumento `A` concluye `P`, y un argumento `B` concluye `¬P`. `B` ataca `A` por refutación.
2.  **Argumento con Premisa Falsa:** Un argumento `A` usa la premisa `Q`, y un argumento `B` concluye `¬Q`. `B` ataca `A` como un defeater.

## Conexiones

### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)

### Técnicas Aplicables
- [[T001]] - Constraint Propagation
- [[T007]] - Formal Concept Analysis

### Conceptos Relacionados
- [[K006]] - Lógica Proposicional
- [[K007]] - Lógica de Predicados
- [[K009]] - Validez Lógica

## Recursos

### Literatura Clave
1.  Besnard, P., & Hunter, A. (2008). *Elements of argumentation*. MIT Press.
2.  Dung, P. M. (1995). *On the acceptability of arguments and its fundamental role in nonmonotonic reasoning, logic programming and n-person games*. Artificial Intelligence, 77(2), 321-358.

## Estado de Implementación

- [x] Investigación completada
- [ ] Diseño de mapeo
- [ ] Implementación
- [ ] Visualización
- [ ] Documentación
- [ ] Tutorial

## Notas Adicionales

Estos marcos son fundamentales para el desarrollo de sistemas de razonamiento automático que necesitan entender la estructura interna de los argumentos.


### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)


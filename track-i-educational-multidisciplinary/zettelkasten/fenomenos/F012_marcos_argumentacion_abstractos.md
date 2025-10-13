---
id: F012
tipo: fenomeno
titulo: Marcos de Argumentación Abstractos (Dung's Framework)
dominios: [logica, inteligencia_artificial, filosofia]
categorias: [C007] # Sistemas de Razonamiento
tags: [argumentacion, dung, afs, semanticas, aceptacion, conflicto]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo
prioridad: alta
---

# Marcos de Argumentación Abstractos (Dung's Framework)

## Descripción

Los **Marcos de Argumentación Abstractos** (Abstract Argumentation Frameworks, AFs), propuestos por Phan Minh Dung en 1995, son una forma fundamental de modelar la argumentación [16]. Un AF se define formalmente como un par `AF = (A, R)`, donde `A` es un conjunto de argumentos abstractos y `R` es una relación binaria sobre `A` que representa la relación de ataque (`a` ataca a `b`) [16]. En este modelo, los argumentos son entidades atómicas sin estructura interna; la clave reside en las relaciones de ataque entre ellos. A partir de estas relaciones, se definen diversas **semánticas de aceptación** que determinan qué conjuntos de argumentos pueden ser aceptados conjuntamente [16, 18].

## Componentes Clave

-   **Argumentos (A):** Entidades atómicas sin estructura interna, representadas como nodos en un grafo dirigido.
-   **Relación de Ataque (R):** Relación binaria entre argumentos, representada como aristas dirigidas en el grafo.
-   **Semánticas de Aceptación:** Criterios para determinar la aceptabilidad de argumentos o conjuntos de argumentos (Conflict-Free, Admissible, Complete, Preferred, Stable, Grounded).

## Mapeo a Formalismos

### CSP

Los AFs de Dung se pueden mapear directamente a CSPs:

-   **Variables:** Cada argumento `a ∈ A` puede ser una variable en el CSP. El dominio de cada variable sería `{aceptado, rechazado, indecidido}`.
-   **Restricciones:** Las relaciones de ataque `(a, b) ∈ R` se traducen en restricciones. Por ejemplo, si `a` ataca `b`, y `a` es `aceptado`, entonces `b` debe ser `rechazado`. El `AdaptiveConsistencyEngine` de LatticeWeaver puede encontrar asignaciones consistentes de estados para todos los argumentos, correspondiendo a las extensiones del AF.

### FCA

Aunque menos directo que con CSP, FCA podría usarse para clasificar argumentos o conjuntos de argumentos basados en propiedades de sus semánticas de aceptación (ej., si pertenecen a una extensión estable, preferida, etc.).

## Ejemplos Concretos

1.  **Ciclo de Ataques:** Tres argumentos A, B, C donde A ataca B, B ataca C, y C ataca A. En este caso, no hay extensiones estables.
2.  **Argumento con Defensor:** A ataca B, C ataca A. C defiende a B de A.

## Conexiones

### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)

### Isomorfismos
- [[I008]] - Isomorfismo entre AFs y CSP (a crear)

### Técnicas Aplicables
- [[T001]] - Constraint Propagation

### Conceptos Relacionados
- [[K007]] - Lógica de Predicados
- [[K009]] - Validez Lógica
- [[K010]] - Semánticas de Aceptación (a crear)

## Recursos

### Literatura Clave
1.  Dung, P. M. (1995). *On the acceptability of arguments and its fundamental role in nonmonotonic reasoning, logic programming and n-person games*. Artificial Intelligence, 77(2), 321-358.
2.  Caminada, M. (s.f.). *An introduction to argumentation semantics*. Recuperado de https://mysite.cs.cf.ac.uk/CaminadaM/publications/KER-BaroniCaminadaGiacomin.pdf

## Estado de Implementación

- [x] Investigación completada
- [ ] Diseño de mapeo
- [ ] Implementación
- [ ] Visualización
- [ ] Documentación
- [ ] Tutorial

## Notas Adicionales

Esta nota se centra en la estructura abstracta de los AFs. Las semánticas de aceptación se detallarán en una nota de concepto separada.


### Conceptos Inversos
- [[K010]] - Semánticas de Aceptación (Argumentación)



### Fenómenos Inversos
- [[F014]] - Marcos de Argumentación Basados en Valores (VAFs)


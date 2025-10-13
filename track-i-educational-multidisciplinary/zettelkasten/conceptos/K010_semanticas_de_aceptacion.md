---
id: K010
tipo: concepto
titulo: Semánticas de Aceptación (Argumentación)
dominios: [logica, inteligencia_artificial, filosofia]
categorias: [C007] # Sistemas de Razonamiento
tags: [argumentacion, dung, afs, aceptacion, conflicto, extensiones]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo
prioridad: alta
---

# Semánticas de Aceptación (Argumentación)

## Descripción

Las **semánticas de aceptación** en los Marcos de Argumentación Abstractos (AFs) de Dung definen los criterios bajo los cuales un argumento o un conjunto de argumentos pueden ser considerados aceptables dentro de un sistema de argumentación [16, 18]. Estas semánticas permiten clasificar los argumentos según su grado de aceptabilidad y la robustez de su defensa, y la elección de una semántica u otra depende del contexto y de los requisitos específicos del razonamiento.

## Tipos de Semánticas

Las principales semánticas son:

-   **Conflict-Free (Libre de Conflictos):** Un conjunto de argumentos `S` es libre de conflictos si no hay dos argumentos en `S` que se ataquen mutuamente. Es una condición básica para la coherencia interna.

-   **Admissible (Admisible):** Un conjunto libre de conflictos `S` es admisible si defiende todos sus argumentos. Para cada argumento `a` en `S`, y para cada argumento `b` que ataca `a`, existe un argumento `c` en `S` que ataca `b`.

-   **Complete (Completa):** Un conjunto admisible `S` es completo si contiene todos los argumentos que defiende. Si un argumento `a` es defendido por `S`, entonces `a` debe estar en `S`.

-   **Preferred (Preferida):** Una extensión preferida es un conjunto admisible maximal (con respecto a la inclusión de conjuntos). Representa un punto de vista coherente y defendible que es lo más amplio posible.

-   **Stable (Estable):** Una extensión estable es un conjunto libre de conflictos `S` que ataca a todos los argumentos que no pertenecen a `S`. Las extensiones estables son siempre preferidas, pero no todas las preferidas son estables.

-   **Grounded (Fundamentada):** La extensión fundamentada es el conjunto completo más pequeño (con respecto a la inclusión de conjuntos). Representa los argumentos que son aceptables de manera irrefutable, es decir, aquellos que pueden ser defendidos sin depender de argumentos que a su vez necesiten ser defendidos por ellos mismos. Es única y siempre existe.

## Mapeo a Formalismos

### CSP

El cálculo de las diferentes semánticas puede mapearse a la resolución de CSPs. Por ejemplo, encontrar una extensión estable implica buscar una asignación de estados (aceptado/rechazado) a los argumentos que satisfaga las condiciones de conflicto-free y ataque a los argumentos externos. El `AdaptiveConsistencyEngine` de LatticeWeaver puede ser fundamental para computar estas extensiones.

## Conexiones

### Instancia de
- [[F012]] - Marcos de Argumentación Abstractos (Dung's Framework)

### Conceptos Relacionados
- [[K009]] - Validez Lógica
- [[K008]] - Inferencia

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

Esta nota detalla las semánticas de aceptación, un concepto central en la teoría de la argumentación abstracta.


### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)
- [[F012]] - Marcos de Argumentación Abstractos (Dung's Framework)



### Instancia de
- [[F014]] - Marcos de Argumentación Basados en Valores (VAFs)



### Instancia de
- [[F015]] - Marcos de Argumentación Basados en Supuestos (ABAs)



### Conceptos Inversos
- [[K018]] - Lógica Dialógica y Juegos de Diálogo



### Categorías Inversas
- [[C007]] - Sistemas de Razonamiento



### Isomorfismos Inversos
- [[I008]] - Isomorfismo entre Marcos de Argumentación Abstractos (AFs) y CSP



### Isomorfismos Inversos
- [[I008]] - Isomorfismo entre Marcos de Argumentación Abstractos (AFs) y CSP



### Fenómenos Inversos
- [[F011]] - Lógica y Argumentación (Filosofía)



### Conceptos Inversos
- [[K009]] - Validez Lógica



### Conceptos Inversos
- [[K011]] - Valores y Preferencias



### Fenómenos Inversos
- [[F011]] - Lógica y Argumentación (Filosofía)



### Conceptos Inversos
- [[K009]] - Validez Lógica



### Fenómenos Inversos
- [[F011]] - Lógica y Argumentación (Filosofía)



### Conceptos Inversos
- [[K009]] - Validez Lógica


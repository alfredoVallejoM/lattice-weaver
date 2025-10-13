---
id: K016
tipo: concepto
titulo: Lógica Paraconsistente
dominios: [logica, filosofia, inteligencia_artificial]
categorias: [C006] # Satisfacibilidad Lógica
tags: [inconsistencia, contradiccion, no_trivialidad, razonamiento_inconsistente]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo
prioridad: media
---

# Lógica Paraconsistente

## Descripción

La **Lógica Paraconsistente** es una lógica no clásica que permite que un sistema de conocimiento contenga contradicciones sin que se vuelva trivial, es decir, sin que de ellas se pueda deducir cualquier proposición [27]. En la lógica clásica, una sola contradicción (`P AND NOT P`) implica cualquier cosa, lo que la hace inútil para razonar con información inconsistente. La lógica paraconsistente es crucial para modelar bases de conocimiento con información contradictoria, razonamiento legal donde existen leyes en conflicto, o sistemas de creencias humanas que a menudo son inconsistentes.

## Componentes Clave

-   **Principio de No Trivialidad:** Una contradicción no implica todas las proposiciones.
-   **Tolerancia a la Inconsistencia:** Permite la presencia de contradicciones controladas.

## Mapeo a Formalismos

### CSP

-   **Variables:** Proposiciones que pueden ser inconsistentes.
-   **Dominios:** `{Verdadero, Falso, Ambos, Ninguno}` (para lógicas de cuatro valores) o grados de verdad en lógicas difusas paraconsistentes.
-   **Restricciones:** Las reglas de inferencia se modifican para no permitir la explosión lógica. Por ejemplo, `P AND NOT P` no implicaría `Q`. El `AdaptiveConsistencyEngine` de LatticeWeaver podría adaptarse para manejar dominios de variables extendidos y reglas de propagación de restricciones que respeten la paraconsistencia.

## Conexiones

### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)

### Conceptos Relacionados
- [[K006]] - Lógica Proposicional
- [[K009]] - Validez Lógica

## Recursos

### Literatura Clave
1.  Priest, G. (2002). *Paraconsistent Logic*. Stanford Encyclopedia of Philosophy.

## Estado de Implementación

- [x] Investigación completada
- [ ] Diseño de mapeo
- [ ] Implementación
- [ ] Visualización
- [ ] Documentación
- [ ] Tutorial

## Notas Adicionales

La lógica paraconsistente es fundamental para el desarrollo de sistemas de IA robustos que operan en entornos con información incierta o contradictoria.


### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)


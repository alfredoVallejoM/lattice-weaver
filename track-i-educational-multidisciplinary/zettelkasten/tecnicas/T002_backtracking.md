---
id: T002
tipo: tecnica
titulo: Backtracking
dominios: [inteligencia_artificial, informatica, optimizacion]
categorias: [C006] # Satisfacibilidad Lógica
tags: [busqueda, csp, algoritmos, satisfacibilidad]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo
prioridad: media
---

# Backtracking

## Descripción

El **Backtracking** es un algoritmo general para encontrar todas (o algunas) soluciones a algunos problemas computacionales, notablemente problemas de satisfacción de restricciones (CSP) y problemas de optimización. Construye candidatos a soluciones de forma incremental, y abandona un candidato ("backtracks") tan pronto como determina que el candidato no puede ser completado a una solución válida. Es una técnica de búsqueda exhaustiva que explora el espacio de estados de forma sistemática [1].

## Componentes Clave

-   **Función de Búsqueda:** Explora el espacio de soluciones.
-   **Función de Poda:** Elimina ramas del árbol de búsqueda que no pueden llevar a una solución.
-   **Asignación Parcial:** Solución incompleta que se extiende incrementalmente.

## Mapeo a Formalismos

### CSP

El backtracking es un algoritmo fundamental para resolver CSPs. A menudo se combina con la propagación de restricciones (`[[T001]]`) para mejorar su eficiencia, permitiendo podar el espacio de búsqueda de manera más agresiva. El `AdaptiveConsistencyEngine` de LatticeWeaver utiliza principios de backtracking y propagación para encontrar soluciones a los problemas de restricciones.

## Conexiones

### Instancia de
- [[C006]] - Satisfacibilidad Lógica

### Técnicas Aplicables
- [[T001]] - Constraint Propagation

### Conceptos Relacionados
- [[K008]] - Inferencia

## Recursos

### Literatura Clave
1.  Russell, S. J., & Norvig, P. (2010). *Artificial Intelligence: A Modern Approach*. Prentice Hall.

## Estado de Implementación

- [x] Investigación completada
- [ ] Diseño de mapeo
- [ ] Implementación
- [ ] Visualización
- [ ] Documentación
- [ ] Tutorial

## Notas Adicionales

El backtracking es un algoritmo versátil, pero su eficiencia puede variar drásticamente dependiendo de la heurística de ordenación de variables y valores, y de la potencia de la propagación de restricciones.

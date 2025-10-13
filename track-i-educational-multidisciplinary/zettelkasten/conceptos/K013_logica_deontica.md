---
id: K013
tipo: concepto
titulo: Lógica Deóntica
dominios: [logica, filosofia, etica, derecho]
categorias: [C006] # Satisfacibilidad Lógica
tags: [obligacion, prohibicion, permiso, normas, moral, etica, derecho]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo
prioridad: media
---

# Lógica Deóntica

## Descripción

La **lógica deóntica** es una lógica modal especializada que formaliza conceptos normativos como la obligación (O), la prohibición (P) y el permiso (F) [24]. Es fundamental en la argumentación ética y jurídica, donde se discuten deberes, derechos y acciones permitidas o prohibidas. Permite analizar la coherencia de sistemas normativos y la validez de argumentos que se basan en principios morales o legales.

## Componentes Clave

-   **Operadores Deónticos:** O (obligatorio), P (prohibido), F (permitido).
-   **Normas:** Proposiciones que expresan obligaciones, prohibiciones o permisos.
-   **Sistemas Normativos:** Conjuntos de normas y sus interrelaciones.

## Mapeo a Formalismos

### CSP

-   **Variables:** Proposiciones que describen acciones o estados de cosas. También se pueden introducir variables para representar la validez o aplicabilidad de una norma.
-   **Dominios:** `{Verdadero, Falso}` para las proposiciones; `{aplicable, no_aplicable}` para las normas.
-   **Restricciones:** Las relaciones entre los operadores deónticos y las proposiciones se modelan como restricciones. Por ejemplo, si una acción `A` es obligatoria, entonces `A` debe ser `Verdadero` en un mundo ideal. Si `A` está prohibida, entonces `A` debe ser `Falso`. Las inconsistencias normativas (ej., una acción es obligatoria y prohibida a la vez) se pueden detectar como CSPs insatisfacibles.

## Conexiones

### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)
- [[K012]] - Lógica Modal

### Conceptos Relacionados
- [[K009]] - Validez Lógica
- [[K011]] - Valores y Preferencias

## Recursos

### Literatura Clave
1.  Von Wright, G. H. (1951). *Deontic Logic*. Mind, 60(237), 1-15.

## Estado de Implementación

- [x] Investigación completada
- [ ] Diseño de mapeo
- [ ] Implementación
- [ ] Visualización
- [ ] Documentación
- [ ] Tutorial

## Notas Adicionales

La lógica deóntica es esencial para el razonamiento práctico y la inteligencia artificial en dominios legales y éticos.


### Conceptos Inversos
- [[K012]] - Lógica Modal


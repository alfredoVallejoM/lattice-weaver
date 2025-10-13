---
id: K017
tipo: concepto
titulo: Lógica Relevante
dominios: [logica, filosofia, inteligencia_artificial]
categorias: [C006] # Satisfacibilidad Lógica
tags: [relevancia, implicacion, paradojas, logica_clasica]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo
prioridad: media
---

# Lógica Relevante

## Descripción

La **Lógica Relevante** es una lógica no clásica que busca abordar las "paradojas de la implicación material" presentes en la lógica clásica [28]. En la lógica clásica, una proposición falsa implica cualquier cosa, y una proposición verdadera es implicada por cualquier cosa. La lógica relevante exige que, para que una implicación `A -> B` sea verdadera, debe haber una conexión de **relevancia** entre `A` y `B`. Esto significa que `A` debe ser realmente "relevante" para `B`. Es fundamental para modelar el razonamiento humano, donde la relevancia es un criterio intuitivo para la validez de un argumento.

## Componentes Clave

-   **Principio de Relevancia:** La verdad de una implicación requiere una conexión temática o conceptual entre el antecedente y el consecuente.
-   **Evitar Paradojas:** Resuelve las paradojas de la implicación material y la implicación estricta.

## Mapeo a Formalismos

### CSP

-   **Variables:** Proposiciones.
-   **Dominios:** `{Verdadero, Falso}`.
-   **Restricciones:** Las restricciones para la implicación `A -> B` no solo considerarían los valores de verdad, sino también la "relevancia" entre `A` y `B`. Esto podría implicar la introducción de variables adicionales para la relevancia o la modificación de las reglas de propagación de restricciones para asegurar que solo se infieran conclusiones relevantes. El `AdaptiveConsistencyEngine` de LatticeWeaver necesitaría extensiones para manejar esta noción de relevancia.

## Conexiones

### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)

### Conceptos Relacionados
- [[K006]] - Lógica Proposicional
- [[K009]] - Validez Lógica

## Recursos

### Literatura Clave
1.  Anderson, A. R., & Belnap, N. D. (1975). *Entailment: The Logic of Relevance and Necessity*. Princeton University Press.

## Estado de Implementación

- [x] Investigación completada
- [ ] Diseño de mapeo
- [ ] Implementación
- [ ] Visualización
- [ ] Documentación
- [ ] Tutorial

## Notas Adicionales

La lógica relevante es un área activa de investigación que busca una formalización más intuitiva de la implicación lógica.


### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)


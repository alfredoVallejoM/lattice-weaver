# Desafíos y Limitaciones de la Implementación de las Técnicas de Andrews en CbO

Durante la implementación de las Técnicas 2 y 3 de Andrews para la optimización del algoritmo Close-by-One (CbO) en el proyecto LatticeWeaver, se encontraron desafíos significativos, principalmente relacionados con las **pruebas de canonicidad**.

## Problema Identificado: Fallos en las Pruebas de Canonicidad

Las optimizaciones propuestas por Andrews buscan reducir el espacio de búsqueda y el número de conceptos generados, mejorando así la eficiencia del algoritmo CbO. Sin embargo, la integración de estas técnicas en la implementación actual de `LatticeBuilder` resultó en fallos en las pruebas de canonicidad. Esto significa que los conceptos generados con las optimizaciones no siempre eran únicos o correctos, lo que llevaba a retículos de conceptos incompletos o incorrectos.

### Causas Probables de los Fallos:

*   **Interpretación Incorrecta de las Condiciones de Canonicidad:** Las condiciones para determinar la canonicidad de un concepto en el contexto de las optimizaciones de Andrews son sutiles y complejas. Es posible que la implementación no haya capturado todas las sutilezas necesarias para garantizar la corrección.
*   **Interacción con la Estructura Existente del CbO:** La implementación base del algoritmo CbO en `LatticeBuilder` ya maneja ciertas lógicas de generación y verificación. La introducción de las nuevas técnicas pudo haber creado conflictos o redundancias que afectaron la validez de los conceptos.
*   **Manejo de Atributos y Objetos Ordenados:** Las Técnicas de Andrews a menudo dependen de un orden específico de objetos y atributos para sus pruebas de canonicidad. Cualquier desviación en este orden o en la forma en que se manejan los `prime_objects` o `prime_attributes` podría invalidar las optimizaciones.

## Consecuencias de los Fallos

La principal consecuencia de estos fallos fue la **inestabilidad y la incorrección** en la construcción del retículo de conceptos. Dado que la precisión del retículo es fundamental para el análisis de conceptos formales, se tomó la decisión de revertir estas optimizaciones para asegurar la funcionalidad y estabilidad del módulo `LatticeBuilder`.

## Decisión y Próximos Pasos

Se ha decidido **revertir las optimizaciones de Andrews** en `builder.py` para restaurar la funcionalidad correcta del algoritmo CbO. Esto garantiza que el proyecto mantenga una base sólida y funcional para futuras mejoras.

Para abordar este problema en el futuro, se recomienda:

1.  **Revisión Detallada del Artículo Original:** Una relectura exhaustiva del artículo de Andrews, prestando especial atención a los ejemplos y las condiciones formales de canonicidad.
2.  **Implementación Incremental y Aislada:** Desarrollar las optimizaciones en un entorno más controlado, con pruebas unitarias específicas para cada aspecto de las Técnicas de Andrews, antes de integrarlas completamente en el algoritmo principal.
3.  **Consulta a Expertos:** Si es posible, buscar la opinión de expertos en FCA o en la implementación de algoritmos de generación de retículos.

Esta documentación sirve como un registro de los desafíos encontrados y la justificación detrás de la decisión de revertir las optimizaciones, permitiendo un enfoque más estructurado para futuras implementaciones.

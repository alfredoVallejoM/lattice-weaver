# Directorio de Análisis de Benchmarks de LatticeWeaver

Este directorio contiene todos los artefactos generados durante el análisis de rendimiento del compilador multiescala **LatticeWeaver**.

## Contenido del Directorio

A continuación se describe el contenido de cada archivo en este directorio:

### Informes

-   `final_report.md`: El informe final consolidado que resume todo el proceso de benchmarking, los resultados, el análisis y las recomendaciones.
-   `benchmark_report.md`: Un informe estadístico detallado generado a partir de los resultados de los benchmarks, con tablas de métricas de rendimiento.
-   `state_of_the_art_solvers.md`: Un documento de investigación que resume los hallazgos sobre solvers CSP del estado del arte, sus técnicas y su rendimiento comparativo.
-   `csplib_problems.md`: Notas sobre problemas de benchmark estándar de la biblioteca CSPLib, utilizados como referencia para la suite de benchmarks.

### Visualizaciones

-   `performance_comparison.png`: Un gráfico que compara el rendimiento general (tiempo total, tasa de éxito, uso de memoria) de las diferentes estrategias de compilación.
-   `scalability_analysis.png`: Un gráfico que analiza la escalabilidad de las diferentes estrategias en varios tipos de problemas y tamaños.
-   `compilation_overhead.png`: Un gráfico que muestra la mejora (o empeoramiento) del rendimiento de cada estrategia de compilación en comparación con la estrategia sin compilación.

### Datos

-   `statistics.json`: Un archivo JSON que contiene los datos estadísticos brutos calculados a partir de los resultados de los benchmarks.

## Cómo Utilizar este Directorio

1.  **Comience con `final_report.md`**: Este informe proporciona una visión general completa del proyecto y sus conclusiones.
2.  **Consulte los informes detallados**: Para un análisis más profundo, revise `benchmark_report.md` y `state_of_the_art_solvers.md`.
3.  **Examine las visualizaciones**: Los archivos PNG proporcionan una representación gráfica de los resultados y son útiles para comprender rápidamente las tendencias de rendimiento.
4.  **Acceda a los datos brutos**: El archivo `statistics.json` está disponible para quienes deseen realizar su propio análisis de los datos.


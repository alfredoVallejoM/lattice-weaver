## Track I: Educativo Multidisciplinar - Zettelkasten

Este directorio contiene la implementación del sistema Zettelkasten para el Track I, enfocado en el mapeo educativo multidisciplinar de fenómenos complejos.

### Estructura de Contenidos

- `zettelkasten/`: Contiene todas las notas del Zettelkasten, organizadas por tipo (fenómenos, categorías, isomorfismos, técnicas, dominios, conceptos, mapeos).
- `scripts/zettelkasten/`: Scripts Python para la gestión del Zettelkasten, incluyendo:
    - `create_note.py`: Script para la creación automatizada de nuevas notas.
    - `validate_zettelkasten.py`: Script para la validación de la consistencia y los enlaces bidireccionales de las notas.
    - `update_catalog.py`: Script para la actualización del `CATALOGO_MAESTRO.md`.
    - `generate_graph.py`: Script para la generación del grafo de visualización interactivo.
- `templates/`: Plantillas para la creación de notas y otros archivos generados.
- `docs/`: Documentación específica del Track I, incluyendo:
    - `CATALOGO_MAESTRO.md`: Catálogo maestro de todas las notas del Zettelkasten.
    - `visualizations/zettelkasten_graph.html`: Visualización interactiva del grafo del Zettelkasten.
- `investigacion/`: Documentos de investigación y análisis de fenómenos.
- `planes/`: Planes de trabajo y desarrollo del Track I.

### Archivos Generados

- `CATALOGO_MAESTRO.md`: Generado por `update_catalog.py`, lista todas las notas con sus metadatos.
- `zettelkasten_graph.html`: Generado por `generate_graph.py`, proporciona una visualización interactiva de las relaciones entre las notas.

### Uso

Para interactuar con el Zettelkasten, se recomienda utilizar los scripts proporcionados en `scripts/zettelkasten/`.

Para ver el grafo de visualización, abre el archivo `docs/visualizations/zettelkasten_graph.html` en tu navegador web.


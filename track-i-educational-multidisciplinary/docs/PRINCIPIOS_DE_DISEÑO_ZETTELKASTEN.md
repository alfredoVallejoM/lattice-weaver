# Principios de Diseño del Zettelkasten para LatticeWeaver (Track I)

Este documento establece los principios fundamentales que guiarán la creación, organización y mantenimiento del Zettelkasten del Track I de LatticeWeaver. Estos principios aseguran la coherencia, la calidad y la utilidad a largo plazo de nuestra base de conocimiento interdisciplinar.

## 1. Atomicidad y Unicidad de la Nota

-   **Un concepto, una nota:** Cada nota debe centrarse en una única idea, concepto, fenómeno, isomorfismo o técnica. Si una nota se vuelve demasiado extensa o abarca múltiples ideas, debe dividirse en notas más pequeñas y enlazadas.
-   **ID Único:** Cada nota tendrá un identificador único (ej. F001, C002, I003) que no cambiará. Este ID es la referencia principal de la nota.
-   **Autocontenido:** Una nota debe ser comprensible por sí misma, aunque su valor se maximice a través de sus conexiones.

## 2. Interconexión Explícita y Tipada

-   **Enlaces Bidireccionales:** Siempre que se haga referencia a otra nota, se debe establecer un enlace bidireccional. Esto significa que si la Nota A enlaza a la Nota B, la Nota B debe tener un enlace inverso a la Nota A.
-   **Enlaces Tipados:** Los enlaces deben indicar la naturaleza de la relación (ej. `[[F001]] - instancia`, `[[C001]] - categoría`, `[[I001]] - isomorfismo`). Esto enriquece el grafo de conocimiento y permite búsquedas semánticas.
-   **Contexto del Enlace:** Los enlaces deben estar incrustados en el texto de la nota, proporcionando contexto y explicando por qué la conexión es relevante.

## 3. Formato Estandarizado y Metadatos Ricos

-   **Templates Obligatorios:** Todas las notas deben crearse utilizando los templates predefinidos (`template_fenomeno.md`, `template_isomorfismo.md`, etc.). Esto asegura una estructura consistente.
-   **Metadatos YAML:** Cada nota debe comenzar con un bloque YAML (front matter) que contenga metadatos esenciales como `id`, `tipo`, `titulo`, `dominios`, `categorias`, `tags`, `fecha_creacion`, `fecha_modificacion`, `estado` y `prioridad`.
-   **Nomenclatura Consistente:** Los nombres de archivo deben seguir el formato `ID_titulo_descriptivo.md` (ej. `F001_teoria_de_juegos_evolutiva.md`).

## 4. Rigor Científico y Referenciación

-   **Fuentes Confiables:** Toda la información fáctica y las afirmaciones deben estar respaldadas por referencias a literatura científica, libros de texto o fuentes académicas creíbles.
-   **Citas Explícitas:** Las referencias deben citarse explícitamente dentro del texto de la nota y listarse en una sección de "Referencias Clave".
-   **Validación de Isomorfismos:** Los isomorfismos deben ser justificados con un "Nivel de Isomorfismo" (Exacto, Fuerte, Débil) y un "Mapeo Estructural" detallado que demuestre la correspondencia entre componentes y relaciones.

## 5. Escalabilidad y Mantenibilidad

-   **Automatización:** Se deben utilizar los scripts de automatización (`create_note.py`, `update_catalog.py`, `validate_zettelkasten.py`, etc.) para mantener la consistencia y facilitar la gestión del Zettelkasten.
-   **Catálogo Maestro:** El `CATALOGO_MAESTRO.md` debe ser regenerado regularmente para reflejar el estado actual del Zettelkasten y servir como índice principal.
-   **Validación Continua:** El script de validación debe ejecutarse periódicamente para identificar y corregir enlaces rotos, notas huérfanas o inconsistencias en los metadatos.

## 6. Evolución y Adaptabilidad

-   **Estado de la Nota:** Cada nota debe tener un `estado` (borrador, en_revision, completo) que refleje su madurez. Las notas pueden evolucionar con el tiempo.
-   **Priorización:** La `prioridad` de una nota ayuda a enfocar los esfuerzos de desarrollo y refinamiento.
-   **Flexibilidad:** Aunque los principios son importantes, deben permitir la flexibilidad necesaria para incorporar nuevos tipos de conocimiento o adaptar la estructura si es necesario, siempre documentando los cambios.

Estos principios son la base para construir un Zettelkasten que no solo almacene información, sino que también fomente el descubrimiento de nuevas conexiones y la comprensión profunda de los fenómenos interdisciplinares en LatticeWeaver.

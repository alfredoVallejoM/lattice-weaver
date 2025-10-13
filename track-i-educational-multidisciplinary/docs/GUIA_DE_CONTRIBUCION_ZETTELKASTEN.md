# Guía de Contribución al Zettelkasten de LatticeWeaver (Track I)

Esta guía proporciona instrucciones detalladas para todos los colaboradores que deseen añadir o modificar contenido en el Zettelkasten del Track I. Seguir estas directrices es crucial para mantener la coherencia, la calidad y la utilidad de nuestra base de conocimiento interdisciplinar.

## 1. Flujo de Trabajo General

1.  **Clonar el Repositorio:** Asegúrate de tener la última versión del repositorio `lattice-weaver`.
2.  **Crear una Nueva Rama:** Trabaja siempre en una rama separada para tus contribuciones.
3.  **Crear/Modificar Notas:** Utiliza los scripts y templates para añadir o editar notas.
4.  **Actualizar Catálogo:** Regenera el `CATALOGO_MAESTRO.md`.
5.  **Validar Zettelkasten:** Ejecuta el script de validación para asegurar la consistencia.
6.  **Generar Visualización (Opcional):** Genera el grafo para revisar las conexiones.
7.  **Commit y Push:** Sube tus cambios a tu rama.
8.  **Pull Request:** Abre un Pull Request para que tus cambios sean revisados e integrados.

## 2. Creación de Nuevas Notas

Utiliza el script `create_note.py` para crear nuevas notas. Esto asegura que se utilicen los templates correctos y se genere un ID único.

```bash
cd track-i-educational-multidisciplinary
python3 scripts/zettelkasten/create_note.py
```

El script te pedirá:
-   **Tipo de nota:** `fenomeno`, `isomorfismo`, `categoria`, `tecnica`, `concepto`, `dominio`, `mapeo`.
-   **Título:** Un título descriptivo para la nota.
-   **Dominios:** Lista de dominios separados por comas (ej. `biologia, fisica, informatica`).
-   **Categorías:** Lista de IDs de categorías estructurales separadas por comas (ej. `C001,C004`).
-   **Tags:** Lista de palabras clave separadas por comas (ej. `redes, complejidad, simulacion`).
-   **Prioridad:** `maxima`, `alta`, `media`, `baja`.

### Convenciones de Nomenclatura

-   **ID:** Generado automáticamente por el script (ej. `F001`, `I002`, `C003`).
-   **Nombre de archivo:** `ID_titulo_en_snake_case.md` (ej. `F001_teoria_de_juegos_evolutiva.md`).
    -   Convertir el título a minúsculas.
    -   Reemplazar espacios por guiones bajos.
    -   Eliminar caracteres especiales o acentos.

## 3. Estructura de las Notas (Templates)

Cada tipo de nota tiene un template específico en `templates/`. Asegúrate de rellenar todas las secciones relevantes.

### Bloque YAML (Front Matter)

Todas las notas deben comenzar con un bloque YAML con los siguientes campos:

```yaml
---
id: [ID_UNICO]
tipo: [tipo_de_nota]
titulo: [Título Descriptivo]
dominios: [lista_de_dominios]
categorias: [lista_de_ids_de_categorias]
tags: [lista_de_tags]
fecha_creacion: YYYY-MM-DD
fecha_modificacion: YYYY-MM-DD
estado: borrador  # borrador | en_revision | completo
prioridad: media  # maxima | alta | media | baja
---
```

### Secciones de Contenido

-   **Descripción:** Explicación concisa del fenómeno/concepto/isomorfismo.
-   **Componentes Clave:** Variables, dominios, restricciones, función objetivo.
-   **Mapeo a Formalismos:** Cómo se traduce el fenómeno a CSP, FCA, TDA u otros formalismos.
-   **Ejemplos Concretos:** Al menos 3 ejemplos detallados del fenómeno/isomorfismo en diferentes dominios.
-   **Conexiones:**
    -   **Categoría Estructural:** Enlaces a las categorías estructurales a las que pertenece.
    -   **Conexiones Inversas:** Enlaces bidireccionales a otras notas que hacen referencia a esta.
    -   **Isomorfismos:** Enlaces a isomorfismos relevantes.
    -   **Instancias en Otros Dominios:** Enlaces a fenómenos relacionados.
    -   **Técnicas Aplicables:** Enlaces a notas de técnicas.
    -   **Conceptos Fundamentales:** Enlaces a notas de conceptos.
-   **Propiedades Matemáticas:** Complejidad computacional, propiedades estructurales, teoremas.
-   **Visualización:** Tipos de visualización aplicables, componentes reutilizables.
-   **Recursos:** Literatura clave, datasets, implementaciones existentes.
-   **Implementación en LatticeWeaver:** Arquitectura de código, clases base, estado de implementación.
-   **Notas Adicionales:** Ideas para expansión, preguntas abiertas, observaciones.

## 4. Interconexión y Enlaces Bidireccionales

-   **Formato de Enlace:** Utiliza el formato `[[ID_NOTA]]` para enlaces internos. Ejemplo: `[[F001]] - Teoría de Juegos Evolutiva`.
-   **Enlaces Tipados:** Cuando sea posible, añade una breve descripción del tipo de relación junto al enlace (ej. `[[F001]] - instancia`, `[[C001]] - categoría`).
-   **Creación de Enlaces Inversos:** Si añades un enlace de la Nota A a la Nota B, **debes ir a la Nota B y añadir un enlace inverso a la Nota A** en la sección `### Conexiones Inversas`. El script `validate_zettelkasten.py` te ayudará a identificar enlaces unidireccionales.

## 5. Mantenimiento del Zettelkasten

### 5.1. Actualización del Catálogo Maestro

Después de añadir o modificar cualquier nota, es **obligatorio** actualizar el `CATALOGO_MAESTRO.md`.

```bash
cd track-i-educational-multidisciplinary
python3 scripts/zettelkasten/update_catalog.py
```

### 5.2. Validación de Consistencia

Antes de hacer un commit, ejecuta el script de validación para detectar errores o advertencias.

```bash
cd track-i-educational-multidisciplinary
python3 scripts/zettelkasten/validate_zettelkasten.py
```

-   **Errores:** Deben corregirse antes de cualquier Pull Request.
-   **Advertencias:** Deben revisarse y corregirse si es posible (ej. enlaces unidireccionales).

### 5.3. Visualización del Grafo

Para entender mejor la estructura y las conexiones, puedes generar una visualización interactiva del grafo.

```bash
cd track-i-educational-multidisciplinary
python3 scripts/zettelkasten/visualize_graph.py --output zettelkasten_graph.html
```

## 6. Rigor Científico y Referenciación

-   **Citas:** Todas las afirmaciones deben estar respaldadas por referencias. Utiliza un formato consistente (ej. APA, MLA, o un formato simplificado como `Autor(es), Año, Título`).
-   **Sección de Referencias:** Cada nota debe tener una sección `## Referencias Clave` al final, listando todas las fuentes citadas.
-   **Validación de Isomorfismos:** Para las notas de tipo `isomorfismo`, la sección `## Mapeo Estructural` es crítica. Debe detallar la correspondencia entre componentes, relaciones y propiedades de los fenómenos involucrados.

## 7. Control de Versiones (Git)

-   **Commits Atómicos:** Realiza commits pequeños y enfocados, cada uno resolviendo un problema o añadiendo una característica específica.
-   **Mensajes de Commit Claros:** Utiliza mensajes de commit descriptivos que expliquen qué se hizo y por qué.
-   **Pull Requests:** Abre Pull Requests para que tus cambios sean revisados por otros colaboradores. Incluye una descripción clara de lo que has hecho y cualquier consideración especial.

Siguiendo esta guía, aseguraremos que el Zettelkasten de LatticeWeaver crezca como una fuente de conocimiento de alta calidad, interconectada y fácil de mantener para la comunidad interdisciplinar.

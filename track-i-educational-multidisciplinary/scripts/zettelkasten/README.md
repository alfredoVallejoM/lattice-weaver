# Scripts del Zettelkasten - Track I

Este directorio contiene scripts de automatización para gestionar el Zettelkasten del Track I.

---

## 📜 Scripts Disponibles

### 1. `create_note.py` - Crear Nuevas Notas

Asistente para crear notas con IDs automáticos y templates apropiados.

**Uso:**
```bash
# Modo interactivo (recomendado)
python scripts/zettelkasten/create_note.py --type fenomeno --interactive

# Modo rápido
python scripts/zettelkasten/create_note.py --type categoria --titulo "Redes de Interacción"
```

**Opciones:**
- `--type`: Tipo de nota (fenomeno, categoria, isomorfismo, tecnica, dominio, concepto, mapeo)
- `--interactive`: Modo interactivo que pide información paso a paso
- `--titulo`: Título de la nota (modo no interactivo)

**Ejemplo:**
```bash
$ python scripts/zettelkasten/create_note.py --type fenomeno --interactive

=== Creando nueva nota de tipo: FENOMENO ===

ID asignado: F001

Título de la nota: Teoría de Juegos Evolutiva
Dominios (separados por coma): economia, biologia
Categorías (IDs separados por coma): C004
Tags (separados por coma): juegos, equilibrio, evolucion
Prioridad (maxima/alta/media/baja) [media]: maxima

✅ Nota creada exitosamente:
   ID: F001
   Archivo: zettelkasten/fenomenos/F001_teoria_juegos_evolutiva.md
```

---

### 2. `update_catalog.py` - Regenerar Catálogo Maestro

Extrae metadatos de todas las notas y regenera el catálogo maestro con índices múltiples.

**Uso:**
```bash
python scripts/zettelkasten/update_catalog.py
```

**Opciones:**
- `--output`: Ruta del archivo de salida (default: CATALOGO_MAESTRO.md)

**Ejemplo:**
```bash
$ python scripts/zettelkasten/update_catalog.py

🔍 Recolectando notas del Zettelkasten...
   Encontradas 15 notas
📊 Generando estadísticas...
   Total de conexiones: 42
   Densidad: 2.80 conexiones/nota
📝 Generando catálogo...
💾 Guardando catálogo en CATALOGO_MAESTRO.md...
✅ Catálogo maestro actualizado exitosamente!
```

**Qué genera:**
- Índice por ID
- Índice por dominio
- Índice por categoría
- Índice por tags
- Índice por estado
- Grafo de conexiones
- Métricas de progreso

---

### 3. `validate_zettelkasten.py` - Validar Consistencia

Verifica la consistencia del Zettelkasten detectando errores y advertencias.

**Uso:**
```bash
python scripts/zettelkasten/validate_zettelkasten.py
```

**Opciones:**
- `--fix`: Intentar arreglar problemas automáticamente (no implementado aún)
- `--verbose`: Mostrar información detallada

**Ejemplo:**
```bash
$ python scripts/zettelkasten/validate_zettelkasten.py

================================================================================
REPORTE DE VALIDACIÓN DEL ZETTELKASTEN
================================================================================

📊 RESUMEN

Total de problemas encontrados: 5
  - Errores: 1
  - Advertencias: 3
  - Información: 1

Por categoría:
  - LINK_UNIDIRECTIONAL: 3
  - LINK_BROKEN: 1
  - NOTE_ORPHAN: 1

================================================================================
ERRORES (requieren corrección)
================================================================================

❌ [LINK_BROKEN] Enlace roto: [[F001]] → [[F999]] (destino no existe)
   Archivo: zettelkasten/fenomenos/F001_teoria_juegos_evolutiva.md
```

**Validaciones:**
- ✅ Enlaces bidireccionales
- ✅ IDs duplicados
- ✅ Formato de metadatos YAML
- ✅ Notas huérfanas
- ✅ Enlaces rotos
- ✅ Consistencia de nombres de archivo
- ✅ Referencias a categorías

---

### 4. `visualize_graph.py` - Visualizar Grafo de Conexiones

Genera visualizaciones interactivas del grafo de notas.

**Uso:**
```bash
# Generar HTML interactivo (default)
python scripts/zettelkasten/visualize_graph.py --output graph.html

# Generar GraphML para otras herramientas
python scripts/zettelkasten/visualize_graph.py --format graphml --output graph.graphml
```

**Opciones:**
- `--output`: Archivo de salida
- `--format`: Formato (html, graphml)

**Ejemplo:**
```bash
$ python scripts/zettelkasten/visualize_graph.py --output graph.html

🔍 Recolectando notas del Zettelkasten...
   Encontradas 15 notas
   Total de conexiones: 42
📊 Generando visualización en formato html...
✅ Visualización guardada en: graph.html

💡 Abre el archivo en tu navegador para ver el grafo interactivo:
   file:///home/ubuntu/lattice-weaver/track-i-educational-multidisciplinary/graph.html
```

**Características del HTML:**
- 🎨 Nodos coloreados por tipo
- 📏 Tamaño basado en número de conexiones
- 🔍 Zoom y pan interactivo
- 💡 Tooltips con información
- 📸 Exportar como imagen
- 🎮 Controles de física

**Leyenda de colores:**
- 🟢 Verde: Fenómenos
- 🔵 Azul: Categorías
- 🟠 Naranja: Isomorfismos
- 🟣 Morado: Técnicas
- 🔴 Rojo: Dominios
- 🔷 Cian: Conceptos
- 🟤 Marrón: Mapeos

---

### 5. `search_zettelkasten.py` - Búsqueda Avanzada

Búsqueda potente por contenido, tags, metadatos y caminos entre notas.

**Uso:**
```bash
# Buscar por contenido
python scripts/zettelkasten/search_zettelkasten.py --query "juegos"

# Buscar por tag
python scripts/zettelkasten/search_zettelkasten.py --tag equilibrio

# Filtrar por tipo y dominio
python scripts/zettelkasten/search_zettelkasten.py --type fenomeno --domain biologia

# Encontrar camino entre dos notas
python scripts/zettelkasten/search_zettelkasten.py --path F001 F010

# Encontrar notas relacionadas
python scripts/zettelkasten/search_zettelkasten.py --related F001 --distance 2
```

**Opciones:**
- `--query`: Buscar por contenido
- `--tag`: Buscar por tag
- `--type`: Filtrar por tipo de nota
- `--domain`: Filtrar por dominio
- `--category`: Filtrar por categoría (ID)
- `--state`: Filtrar por estado (borrador, en_revision, completo)
- `--path START_ID END_ID`: Encontrar camino entre dos notas
- `--related NOTE_ID`: Encontrar notas relacionadas
- `--distance N`: Distancia máxima para notas relacionadas (default: 2)

**Ejemplos:**

```bash
# Buscar todas las notas sobre "equilibrio"
$ python scripts/zettelkasten/search_zettelkasten.py --query "equilibrio"

✅ Se encontraron 3 resultados:

1. [[F001]] - Teoría de Juegos Evolutiva
   Tipo: fenomeno
   Contextos encontrados:
     • El **equilibrio de Nash** es un concepto central...
     • Estrategias evolutivamente estables (ESS) son equilibrios...

# Encontrar camino entre dos fenómenos
$ python scripts/zettelkasten/search_zettelkasten.py --path F001 F003

🛤️  Buscando camino de [[F001]] a [[F003]]...

✅ Camino encontrado (longitud: 2):

   🎯 [[F001]] - Teoría de Juegos Evolutiva
      └─> [[I003]] - Dilema del Prisionero Multidominio
         └─> 🎯 [[F003]] - Modelo de Ising 2D

# Encontrar notas relacionadas
$ python scripts/zettelkasten/search_zettelkasten.py --related F001 --distance 2

🔗 Buscando notas relacionadas a [[F001]] (distancia máx: 2)...

✅ Notas relacionadas encontradas:

Distancia 1: 5 notas
  • [[C004]] - Sistemas Dinámicos
  • [[I003]] - Dilema del Prisionero Multidominio
  • [[T005]] - Replicator Dynamics
  • [[K002]] - Equilibrio de Nash
  • [[F015]] - Cooperación en Murciélagos Vampiro

Distancia 2: 8 notas
  • [[F002]] - Redes de Regulación Génica
  • [[F003]] - Modelo de Ising 2D
  ...
```

---

## 🔄 Flujo de Trabajo Típico

### Crear una Nueva Nota

```bash
# 1. Crear nota
python scripts/zettelkasten/create_note.py --type fenomeno --interactive

# 2. Editar el archivo generado para completar contenido

# 3. Actualizar catálogo
python scripts/zettelkasten/update_catalog.py

# 4. Validar consistencia
python scripts/zettelkasten/validate_zettelkasten.py

# 5. Visualizar grafo (opcional)
python scripts/zettelkasten/visualize_graph.py --output graph.html
```

### Explorar el Zettelkasten

```bash
# Buscar notas sobre un tema
python scripts/zettelkasten/search_zettelkasten.py --query "redes"

# Ver todas las notas de un dominio
python scripts/zettelkasten/search_zettelkasten.py --domain biologia

# Encontrar conexiones entre conceptos
python scripts/zettelkasten/search_zettelkasten.py --path F001 F010

# Explorar vecindario de una nota
python scripts/zettelkasten/search_zettelkasten.py --related F001
```

### Mantenimiento

```bash
# Validar consistencia regularmente
python scripts/zettelkasten/validate_zettelkasten.py

# Regenerar catálogo después de cambios
python scripts/zettelkasten/update_catalog.py

# Visualizar para detectar clusters
python scripts/zettelkasten/visualize_graph.py --output graph.html
```

---

## 🛠️ Requisitos

Los scripts requieren Python 3.7+ y las siguientes librerías:

```bash
pip install pyyaml
```

Para visualización HTML, no se requieren dependencias adicionales (usa CDN).

---

## 📝 Notas

- Todos los scripts deben ejecutarse desde el directorio raíz del track-i
- Los scripts son idempotentes: pueden ejecutarse múltiples veces sin efectos adversos
- El catálogo maestro se regenera automáticamente, no editar manualmente
- Los IDs son inmutables: una vez asignados, no cambian

---

## 🐛 Solución de Problemas

### "No se encontró bloque YAML front matter"

Asegúrate de que el archivo comienza con:
```yaml
---
id: F001
tipo: fenomeno
...
---
```

### "ID duplicado"

Verifica que no existan dos archivos con el mismo ID. Usa `validate_zettelkasten.py` para detectar duplicados.

### "Enlace roto"

El enlace apunta a una nota que no existe. Corrige el ID o crea la nota faltante.

### "Enlace unidireccional"

Falta el enlace inverso. Añade `[[ID_origen]]` en la nota destino.

---

## 📚 Documentación Adicional

- Ver `ARQUITECTURA_ZETTELKASTEN.md` para detalles de la estructura
- Ver `CATALOGO_MAESTRO.md` para índice completo de notas
- Ver templates en `templates/` para formato de notas

---

**Última actualización:** 2025-10-12


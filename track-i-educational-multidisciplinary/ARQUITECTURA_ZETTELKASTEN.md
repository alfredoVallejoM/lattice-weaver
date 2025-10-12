# Arquitectura Zettelkasten para Track I - LatticeWeaver

**Versión:** 1.0  
**Fecha:** 12 de Octubre, 2025  
**Propósito:** Definir la estructura de organización del conocimiento para el Track I usando principios Zettelkasten adaptados a relaciones interdisciplinares.

---

## 1. Principios Fundamentales

### 1.1 Atomicidad

Cada nota representa **una idea, concepto o relación única**. Esto permite:
- Reutilización máxima
- Referencias precisas
- Evolución independiente

### 1.2 Conectividad

Las notas se conectan mediante **enlaces explícitos** que capturan la naturaleza de la relación:
- `isomorfo_a:` - Isomorfismo estructural
- `instancia_de:` - Relación de categoría
- `tecnica_aplicable:` - Transferencia de técnica
- `similar_a:` - Analogía o similitud
- `prerequisito_de:` - Dependencia conceptual

### 1.3 Autonomía

Cada nota es **autocontenida y comprensible** sin necesidad de contexto externo, pero enriquecida por sus conexiones.

### 1.4 Evolución Orgánica

La estructura **emerge** de las conexiones, no se impone a priori. El índice se actualiza automáticamente.

---

## 2. Estructura de Directorios

```
track-i-educational-multidisciplinary/
├── ARQUITECTURA_ZETTELKASTEN.md          # Este documento
├── VISION_MULTIDISCIPLINAR.md            # Documento de visión original
├── CATALOGO_MAESTRO.md                   # Índice Zettelkasten (generado)
│
├── zettelkasten/                         # Directorio principal de notas
│   │
│   ├── fenomenos/                        # Notas de fenómenos específicos
│   │   ├── F001_teoria_juegos_evolutiva.md
│   │   ├── F002_redes_regulacion_genica.md
│   │   ├── F003_modelo_ising_2d.md
│   │   └── ...
│   │
│   ├── categorias/                       # Notas de categorías estructurales
│   │   ├── C001_redes_interaccion.md
│   │   ├── C002_asignacion_optima.md
│   │   ├── C003_optimizacion_restricciones.md
│   │   └── ...
│   │
│   ├── isomorfismos/                     # Notas de isomorfismos específicos
│   │   ├── I001_coloracion_grafos_frecuencias.md
│   │   ├── I002_ising_redes_sociales.md
│   │   ├── I003_dilema_prisionero_multidominio.md
│   │   └── ...
│   │
│   ├── tecnicas/                         # Notas de técnicas/algoritmos
│   │   ├── T001_constraint_propagation.md
│   │   ├── T002_gale_shapley.md
│   │   ├── T003_simulated_annealing.md
│   │   └── ...
│   │
│   ├── dominios/                         # Notas de dominios científicos
│   │   ├── D001_biologia.md
│   │   ├── D002_economia.md
│   │   ├── D003_fisica.md
│   │   └── ...
│   │
│   ├── conceptos/                        # Notas de conceptos fundamentales
│   │   ├── K001_transicion_fase.md
│   │   ├── K002_equilibrio_nash.md
│   │   ├── K003_emergencia.md
│   │   └── ...
│   │
│   └── mapeos/                           # Notas de mapeos a formalismos
│       ├── M001_csp_mapping.md
│       ├── M002_fca_mapping.md
│       ├── M003_tda_mapping.md
│       └── ...
│
├── investigacion/                        # Documentos de investigación preliminar
│   ├── INV001_constraint_programming_grn.md
│   ├── INV002_teoria_juegos_evolutiva.md
│   └── ...
│
├── planes/                               # Documentos de planificación
│   ├── PLAN_EXPANSION_2025.md
│   ├── PRIORIZACION_FENOMENOS.md
│   └── ...
│
└── templates/                            # Plantillas para nuevas notas
    ├── template_fenomeno.md
    ├── template_isomorfismo.md
    ├── template_tecnica.md
    └── template_categoria.md
```

---

## 3. Sistema de IDs Únicos

### 3.1 Formato de IDs

Cada nota tiene un **ID único alfanumérico** con prefijo que indica su tipo:

- `F###` - Fenómeno (F001, F002, ...)
- `C###` - Categoría estructural (C001, C002, ...)
- `I###` - Isomorfismo (I001, I002, ...)
- `T###` - Técnica/Algoritmo (T001, T002, ...)
- `D###` - Dominio científico (D001, D002, ...)
- `K###` - Concepto fundamental (K001, K002, ...) [K de "Knowledge"]
- `M###` - Mapeo a formalismo (M001, M002, ...)
- `INV###` - Documento de investigación (INV001, INV002, ...)

### 3.2 Reglas de Asignación

1. **Secuencial:** IDs se asignan en orden de creación
2. **Inmutable:** Una vez asignado, un ID nunca cambia
3. **Único:** No se reutilizan IDs de notas eliminadas
4. **Rastreable:** El catálogo maestro mantiene registro de todos los IDs

---

## 4. Estructura de una Nota Zettelkasten

### 4.1 Metadatos (YAML Front Matter)

Cada nota comienza con metadatos en formato YAML:

```yaml
---
id: F001
tipo: fenomeno
titulo: Teoría de Juegos Evolutiva
dominios: [economia, biologia, sociologia]
categorias: [C004]
tags: [juegos, equilibrio, evolucion, cooperacion]
fecha_creacion: 2025-10-12
fecha_modificacion: 2025-10-12
estado: completo  # borrador | en_revision | completo
prioridad: maxima  # maxima | alta | media | baja
---
```

### 4.2 Cuerpo de la Nota

```markdown
# [Título de la Nota]

## Descripción

Descripción concisa del fenómeno/concepto (1-2 párrafos).

## Componentes Clave

- **Variable 1:** Descripción
- **Variable 2:** Descripción
- **Restricciones:** Descripción

## Mapeo a Formalismos

### CSP
- Variables: ...
- Dominios: ...
- Restricciones: ...

### FCA (si aplica)
- Objetos: ...
- Atributos: ...
- Conceptos: ...

## Ejemplos Concretos

1. **Ejemplo 1:** Descripción breve
2. **Ejemplo 2:** Descripción breve

## Conexiones

### Isomorfismos
- [[I003]] - Dilema del Prisionero Multidominio
- [[I007]] - Juegos de Coordinación

### Técnicas Aplicables
- [[T005]] - Replicator Dynamics
- [[T012]] - Evolutionary Stable Strategy (ESS)

### Instancias en Otros Dominios
- [[F015]] - Cooperación en Murciélagos Vampiro (Biología)
- [[F023]] - Cárteles Económicos (Economía)

### Conceptos Relacionados
- [[K002]] - Equilibrio de Nash
- [[K015]] - Estrategia Evolutivamente Estable

## Recursos

### Literatura Clave
1. Maynard Smith, J. (1982). *Evolution and the Theory of Games*.
2. Axelrod, R. (1984). *The Evolution of Cooperation*.

### Implementaciones
- Código: `lattice_weaver/phenomena/evolutionary_games/`
- Tests: `tests/phenomena/test_evolutionary_games.py`

## Estado de Implementación

- [ ] Investigación completada
- [ ] Diseño de mapeo
- [ ] Implementación
- [ ] Visualización
- [ ] Documentación
- [ ] Tutorial

## Notas Adicionales

Observaciones, ideas para expansión, preguntas abiertas.
```

---

## 5. Catálogo Maestro (CATALOGO_MAESTRO.md)

El catálogo maestro es un **índice generado automáticamente** que proporciona múltiples vistas del Zettelkasten:

### 5.1 Secciones del Catálogo

1. **Índice por ID** - Lista completa de todas las notas
2. **Índice por Tipo** - Agrupado por F, C, I, T, D, K, M
3. **Índice por Dominio** - Agrupado por dominio científico
4. **Índice por Categoría** - Agrupado por categoría estructural
5. **Índice por Tags** - Agrupado por etiquetas
6. **Índice por Estado** - Agrupado por estado de implementación
7. **Grafo de Conexiones** - Representación visual de las relaciones
8. **Estadísticas** - Métricas del Zettelkasten

### 5.2 Formato del Catálogo

```markdown
# Catálogo Maestro - Track I Zettelkasten

**Última actualización:** 2025-10-12  
**Total de notas:** 47  
**Conexiones totales:** 156

---

## Índice por ID

### Fenómenos (F)
- [[F001]] - Teoría de Juegos Evolutiva (Economía, Biología)
- [[F002]] - Redes de Regulación Génica (Biología)
- [[F003]] - Modelo de Ising 2D (Física)
- ...

### Categorías (C)
- [[C001]] - Redes de Interacción
- [[C002]] - Asignación Óptima
- ...

### Isomorfismos (I)
- [[I001]] - Coloración de Grafos ≅ Asignación de Frecuencias
- [[I002]] - Modelo de Ising ≅ Redes Sociales
- ...

---

## Índice por Dominio

### Biología
- [[F002]] - Redes de Regulación Génica
- [[F011]] - Plegamiento de Proteínas
- [[F015]] - Redes Neuronales Biológicas
- ...

### Economía
- [[F001]] - Teoría de Juegos Evolutiva
- [[F009]] - Matching Estable
- ...

---

## Grafo de Conexiones (Top 10 Nodos Más Conectados)

1. [[C001]] - Redes de Interacción (23 conexiones)
2. [[F001]] - Teoría de Juegos Evolutiva (18 conexiones)
3. [[T001]] - Constraint Propagation (15 conexiones)
4. [[K002]] - Equilibrio de Nash (14 conexiones)
5. ...

---

## Estadísticas

- **Fenómenos:** 25
- **Categorías:** 6
- **Isomorfismos:** 20
- **Técnicas:** 15
- **Dominios:** 12
- **Conceptos:** 30
- **Mapeos:** 8

**Cobertura por dominio:**
- Biología: 8 fenómenos
- Economía: 5 fenómenos
- Física: 4 fenómenos
- ...

**Estado de implementación:**
- Completo: 3 fenómenos
- En revisión: 5 fenómenos
- Borrador: 17 fenómenos
```

---

## 6. Convenciones de Nomenclatura

### 6.1 Nombres de Archivo

**Formato:** `[ID]_[nombre_descriptivo].md`

**Ejemplos:**
- `F001_teoria_juegos_evolutiva.md`
- `I002_ising_redes_sociales.md`
- `T003_simulated_annealing.md`

**Reglas:**
- Minúsculas
- Guiones bajos para separar palabras
- Sin caracteres especiales (excepto guiones bajos)
- Máximo 50 caracteres (sin contar ID)

### 6.2 Títulos de Nota

**Formato:** Título Case, descriptivo y preciso

**Ejemplos:**
- "Teoría de Juegos Evolutiva"
- "Isomorfismo: Modelo de Ising ≅ Redes Sociales"
- "Técnica: Algoritmo Gale-Shapley"

### 6.3 Tags

**Categorías de tags:**
- **Conceptuales:** `equilibrio`, `emergencia`, `transicion_fase`
- **Metodológicos:** `csp`, `fca`, `optimizacion`
- **Aplicaciones:** `educacion`, `investigacion`, `industria`
- **Estado:** `implementado`, `en_desarrollo`, `planificado`

**Reglas:**
- Minúsculas
- Guiones bajos para separar palabras
- Máximo 3 palabras por tag
- Preferir tags existentes antes de crear nuevos

---

## 7. Tipos de Enlaces y Relaciones

### 7.1 Enlaces Directos

**Sintaxis:** `[[ID]]` o `[[ID|Texto Alternativo]]`

**Ejemplo:**
```markdown
Este fenómeno es isomorfo a [[I003|el Dilema del Prisionero en múltiples dominios]].
```

### 7.2 Enlaces Tipados

Para capturar la **naturaleza de la relación**, usar secciones específicas:

```markdown
## Conexiones

### Isomorfismos
- [[I003]] - Relación de isomorfismo exacto
- [[I007]] - Relación de isomorfismo fuerte

### Instancias
- [[F015]] - Instancia específica en Biología
- [[F023]] - Instancia específica en Economía

### Técnicas Aplicables
- [[T005]] - Técnica transferible desde Física
- [[T012]] - Técnica desarrollada en este contexto

### Prerequisitos
- [[K002]] - Concepto necesario para comprender esta nota
```

### 7.3 Enlaces Bidireccionales

Cuando se crea un enlace de A → B, **debe crearse** el enlace inverso B → A en la nota destino.

**Herramienta:** Script de validación de consistencia (`scripts/validate_zettelkasten.py`)

---

## 8. Flujo de Trabajo

### 8.1 Crear Nueva Nota

1. **Asignar ID:** Consultar `CATALOGO_MAESTRO.md` para el siguiente ID disponible
2. **Usar plantilla:** Copiar template apropiado de `templates/`
3. **Completar metadatos:** YAML front matter
4. **Escribir contenido:** Seguir estructura estándar
5. **Crear enlaces:** Conectar con notas existentes
6. **Actualizar enlaces inversos:** En notas referenciadas
7. **Actualizar catálogo:** Ejecutar `scripts/update_catalog.py`

### 8.2 Modificar Nota Existente

1. **Actualizar fecha de modificación** en metadatos
2. **Mantener consistencia** de enlaces
3. **Actualizar catálogo** si cambian metadatos clave

### 8.3 Descubrir Conexiones

1. **Búsqueda por tags:** Encontrar notas relacionadas
2. **Explorar grafo:** Usar visualizador de conexiones
3. **Consultar índices:** Por dominio, categoría, etc.
4. **Identificar isomorfismos:** Buscar patrones estructurales

---

## 9. Herramientas de Soporte

### 9.1 Scripts de Automatización

**Ubicación:** `scripts/zettelkasten/`

1. **`update_catalog.py`**
   - Regenera `CATALOGO_MAESTRO.md`
   - Extrae metadatos de todas las notas
   - Genera índices y estadísticas

2. **`validate_zettelkasten.py`**
   - Verifica consistencia de enlaces bidireccionales
   - Detecta IDs duplicados
   - Valida formato de metadatos
   - Reporta notas huérfanas (sin conexiones)

3. **`create_note.py`**
   - Asistente interactivo para crear notas
   - Asigna ID automáticamente
   - Sugiere conexiones basadas en tags/dominios

4. **`visualize_graph.py`**
   - Genera visualización del grafo de conexiones
   - Exporta a formatos: HTML interactivo, GraphML, DOT

5. **`search_zettelkasten.py`**
   - Búsqueda avanzada por contenido, tags, metadatos
   - Búsqueda de caminos entre notas
   - Identificación de clusters

### 9.2 Integración con Obsidian (Opcional)

El formato es **compatible con Obsidian**, permitiendo:
- Visualización de grafo nativa
- Búsqueda y navegación
- Plugins de Zettelkasten

**Configuración:** Abrir `track-i-educational-multidisciplinary/` como vault de Obsidian.

---

## 10. Ejemplos de Notas

### 10.1 Ejemplo: Nota de Fenómeno

Ver: `templates/template_fenomeno.md` (a crear)

### 10.2 Ejemplo: Nota de Isomorfismo

Ver: `templates/template_isomorfismo.md` (a crear)

### 10.3 Ejemplo: Nota de Técnica

Ver: `templates/template_tecnica.md` (a crear)

---

## 11. Métricas de Calidad

### 11.1 Métricas de Nota Individual

- **Completitud:** ¿Todas las secciones están completas?
- **Conectividad:** ¿Tiene al menos 3 conexiones?
- **Claridad:** ¿Es comprensible sin contexto externo?
- **Actualidad:** ¿Metadatos actualizados?

### 11.2 Métricas del Zettelkasten

- **Densidad de conexiones:** Conexiones / Notas
- **Cobertura de dominios:** Distribución uniforme
- **Profundidad de isomorfismos:** Número de instancias por isomorfismo
- **Notas huérfanas:** < 5%

**Objetivo Año 1:**
- 50+ notas de fenómenos
- 20+ notas de isomorfismos
- 200+ conexiones totales
- Densidad > 4 conexiones/nota

---

## 12. Evolución y Mantenimiento

### 12.1 Revisión Periódica

**Frecuencia:** Mensual

**Actividades:**
1. Ejecutar `validate_zettelkasten.py`
2. Revisar notas huérfanas
3. Identificar nuevos isomorfismos
4. Actualizar estadísticas

### 12.2 Refactorización

Cuando una nota crece demasiado (>500 líneas), considerar:
- Dividir en sub-notas
- Extraer conceptos a notas separadas
- Mantener nota original como "índice"

### 12.3 Archivo de Notas Obsoletas

Notas que ya no son relevantes se mueven a `zettelkasten/archive/` pero mantienen su ID para referencias históricas.

---

## 13. Próximos Pasos

1. **Crear templates** en `templates/`
2. **Implementar scripts** en `scripts/zettelkasten/`
3. **Migrar documentos existentes** a formato Zettelkasten
4. **Crear primeras 10 notas** (3 fenómenos piloto + categorías + isomorfismos)
5. **Generar catálogo inicial**
6. **Validar flujo de trabajo** con usuario

---

**Esta arquitectura proporciona una base sólida y escalable para capturar el conocimiento interdisciplinar del Track I, permitiendo que las conexiones emerjan orgánicamente mientras mantenemos rigor y consistencia.**


#!/usr/bin/env python3
"""
Script para regenerar el catálogo maestro del Zettelkasten.

Extrae metadatos de todas las notas y genera índices múltiples.

Uso:
    python update_catalog.py
    python update_catalog.py --output custom_catalog.md
"""

import os
import sys
import argparse
import re
import yaml
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Añadir el directorio raíz al path
SCRIPT_DIR = Path(__file__).parent
TRACK_DIR = SCRIPT_DIR.parent.parent
ZETTEL_DIR = TRACK_DIR / "zettelkasten"
CATALOG_PATH = TRACK_DIR / "CATALOGO_MAESTRO.md"


def extract_metadata(filepath: Path) -> Dict:
    """
    Extrae los metadatos YAML de una nota.
    
    Args:
        filepath: Path al archivo de la nota
    
    Returns:
        Diccionario con los metadatos
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar el bloque YAML front matter
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        return {}
    
    yaml_content = match.group(1)
    try:
        metadata = yaml.safe_load(yaml_content)
        # Añadir el path del archivo
        metadata['_filepath'] = str(filepath.relative_to(TRACK_DIR))
        return metadata
    except yaml.YAMLError as e:
        print(f"Error parseando YAML en {filepath}: {e}")
        return {}


def extract_links(filepath: Path) -> List[str]:
    """
    Extrae todos los enlaces [[ID]] de una nota.
    
    Args:
        filepath: Path al archivo de la nota
    
    Returns:
        Lista de IDs enlazados
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar todos los enlaces [[ID]] o [[ID|texto]]
    links = re.findall(r'\[\[([A-Z]\d{3})', content)
    return list(set(links))  # Eliminar duplicados


def collect_all_notes() -> Dict[str, Dict]:
    """
    Recolecta todas las notas del Zettelkasten.
    
    Returns:
        Diccionario {ID: metadata}
    """
    notes = {}
    
    for subdir in ZETTEL_DIR.iterdir():
        if not subdir.is_dir():
            continue
        
        for filepath in subdir.glob("*.md"):
            metadata = extract_metadata(filepath)
            if metadata and 'id' in metadata:
                note_id = metadata['id']
                metadata['_links'] = extract_links(filepath)
                notes[note_id] = metadata
    
    return notes


def generate_statistics(notes: Dict[str, Dict]) -> Dict:
    """
    Genera estadísticas del Zettelkasten.
    
    Args:
        notes: Diccionario de notas
    
    Returns:
        Diccionario con estadísticas
    """
    stats = {
        'total_notes': len(notes),
        'by_type': defaultdict(int),
        'by_domain': defaultdict(int),
        'by_category': defaultdict(int),
        'by_state': defaultdict(int),
        'total_connections': 0,
        'orphan_notes': [],
        'most_connected': [],
    }
    
    # Contar por tipo
    for note_id, metadata in notes.items():
        note_type = metadata.get('tipo', 'unknown')
        stats['by_type'][note_type] += 1
        
        # Contar conexiones
        links = metadata.get('_links', [])
        stats['total_connections'] += len(links)
        
        # Detectar huérfanas
        if len(links) == 0:
            stats['orphan_notes'].append(note_id)
        
        # Contar por dominio
        dominios = metadata.get('dominios', [])
        if isinstance(dominios, list):
            for dominio in dominios:
                stats['by_domain'][dominio] += 1
        
        # Contar por categoría
        categorias = metadata.get('categorias', [])
        if isinstance(categorias, list):
            for categoria in categorias:
                stats['by_category'][categoria] += 1
        
        # Contar por estado
        estado = metadata.get('estado', 'unknown')
        stats['by_state'][estado] += 1
    
    # Nodos más conectados
    connection_counts = [(note_id, len(metadata.get('_links', []))) 
                         for note_id, metadata in notes.items()]
    stats['most_connected'] = sorted(connection_counts, key=lambda x: x[1], reverse=True)[:10]
    
    # Densidad de conexiones
    if stats['total_notes'] > 0:
        stats['density'] = stats['total_connections'] / stats['total_notes']
    else:
        stats['density'] = 0.0
    
    return stats


def generate_catalog(notes: Dict[str, Dict], stats: Dict) -> str:
    """
    Genera el contenido del catálogo maestro.
    
    Args:
        notes: Diccionario de notas
        stats: Estadísticas
    
    Returns:
        Contenido del catálogo en Markdown
    """
    now = datetime.now().strftime("%Y-%m-%d")
    
    catalog = f"""# Catálogo Maestro - Track I Zettelkasten

**Última actualización:** {now}  
**Total de notas:** {stats['total_notes']}  
**Conexiones totales:** {stats['total_connections']}

---

## 📋 Resumen Ejecutivo

Este catálogo es el **índice maestro** del Zettelkasten del Track I. Proporciona múltiples vistas del conocimiento capturado y se regenera automáticamente mediante el script `scripts/zettelkasten/update_catalog.py`.

### Estadísticas Generales

- **Fenómenos (F):** {stats['by_type'].get('fenomeno', 0)}
- **Categorías (C):** {stats['by_type'].get('categoria', 0)}
- **Isomorfismos (I):** {stats['by_type'].get('isomorfismo', 0)}
- **Técnicas (T):** {stats['by_type'].get('tecnica', 0)}
- **Dominios (D):** {stats['by_type'].get('dominio', 0)}
- **Conceptos (K):** {stats['by_type'].get('concepto', 0)}
- **Mapeos (M):** {stats['by_type'].get('mapeo', 0)}

### Métricas de Conectividad

- **Densidad de conexiones:** {stats['density']:.2f} (conexiones por nota)
- **Notas huérfanas:** {len(stats['orphan_notes'])} ({len(stats['orphan_notes'])/max(stats['total_notes'], 1)*100:.1f}%)
"""
    
    if stats['most_connected']:
        top_note = stats['most_connected'][0]
        catalog += f"- **Nodo más conectado:** [[{top_note[0]}]] ({top_note[1]} conexiones)\n"
    else:
        catalog += "- **Nodo más conectado:** N/A\n"
    
    catalog += "\n---\n\n"
    
    # Índice por ID
    catalog += "## 📚 Índice por ID\n\n"
    
    for note_type, type_name in [
        ('fenomeno', 'Fenómenos (F)'),
        ('categoria', 'Categorías (C)'),
        ('isomorfismo', 'Isomorfismos (I)'),
        ('tecnica', 'Técnicas (T)'),
        ('dominio', 'Dominios (D)'),
        ('concepto', 'Conceptos (K)'),
        ('mapeo', 'Mapeos (M)'),
    ]:
        catalog += f"### {type_name}\n\n"
        
        type_notes = [(nid, meta) for nid, meta in notes.items() 
                      if meta.get('tipo') == note_type]
        type_notes.sort(key=lambda x: x[0])
        
        if type_notes:
            for note_id, metadata in type_notes:
                titulo = metadata.get('titulo', 'Sin título')
                dominios = metadata.get('dominios', [])
                if isinstance(dominios, list) and dominios:
                    dominios_str = f" ({', '.join(dominios)})"
                else:
                    dominios_str = ""
                catalog += f"- [[{note_id}]] - {titulo}{dominios_str}\n"
        else:
            catalog += f"*No hay notas de tipo {note_type} aún.*\n"
        
        catalog += "\n"
    
    catalog += "---\n\n"
    
    # Índice por Dominio
    catalog += "## 🌍 Índice por Dominio\n\n"
    
    dominios_conocidos = [
        'biologia', 'economia', 'fisica', 'sociologia', 'informatica',
        'matematicas', 'linguistica', 'neurociencia', 'medicina',
        'filosofia', 'ciencia_politica'
    ]
    
    for dominio in dominios_conocidos:
        catalog += f"### {dominio.replace('_', ' ').title()}\n\n"
        
        domain_notes = []
        for note_id, metadata in notes.items():
            dominios = metadata.get('dominios', [])
            if isinstance(dominios, list) and dominio in dominios:
                domain_notes.append((note_id, metadata))
        
        domain_notes.sort(key=lambda x: x[0])
        
        if domain_notes:
            for note_id, metadata in domain_notes:
                titulo = metadata.get('titulo', 'Sin título')
                catalog += f"- [[{note_id}]] - {titulo}\n"
        else:
            catalog += f"*No hay notas en este dominio aún.*\n"
        
        catalog += "\n"
    
    # Otros dominios
    otros_dominios = set()
    for metadata in notes.values():
        dominios = metadata.get('dominios', [])
        if isinstance(dominios, list):
            for d in dominios:
                if d not in dominios_conocidos:
                    otros_dominios.add(d)
    
    if otros_dominios:
        catalog += "### Otros\n\n"
        for dominio in sorted(otros_dominios):
            domain_notes = []
            for note_id, metadata in notes.items():
                dominios = metadata.get('dominios', [])
                if isinstance(dominios, list) and dominio in dominios:
                    domain_notes.append((note_id, metadata))
            
            if domain_notes:
                catalog += f"**{dominio.replace('_', ' ').title()}:**\n"
                for note_id, metadata in domain_notes:
                    titulo = metadata.get('titulo', 'Sin título')
                    catalog += f"- [[{note_id}]] - {titulo}\n"
                catalog += "\n"
    
    catalog += "---\n\n"
    
    # Índice por Categoría
    catalog += "## 🏗️ Índice por Categoría Estructural\n\n"
    
    categorias_map = {
        'C001': 'Redes de Interacción',
        'C002': 'Asignación Óptima',
        'C003': 'Optimización con Restricciones',
        'C004': 'Sistemas Dinámicos',
        'C005': 'Jerarquías y Taxonomías',
        'C006': 'Satisfacibilidad Lógica',
    }
    
    for cat_id, cat_name in categorias_map.items():
        catalog += f"### {cat_id} - {cat_name}\n\n"
        
        cat_notes = []
        for note_id, metadata in notes.items():
            categorias = metadata.get('categorias', [])
            if isinstance(categorias, list) and cat_id in categorias:
                cat_notes.append((note_id, metadata))
        
        cat_notes.sort(key=lambda x: x[0])
        
        if cat_notes:
            for note_id, metadata in cat_notes:
                titulo = metadata.get('titulo', 'Sin título')
                catalog += f"- [[{note_id}]] - {titulo}\n"
        else:
            catalog += f"*No hay fenómenos en esta categoría aún.*\n"
        
        catalog += "\n"
    
    catalog += "---\n\n"
    
    # Índice por Tags
    catalog += "## 🏷️ Índice por Tags\n\n"
    
    tags_index = defaultdict(list)
    for note_id, metadata in notes.items():
        tags = metadata.get('tags', [])
        if isinstance(tags, list):
            for tag in tags:
                tags_index[tag].append((note_id, metadata.get('titulo', 'Sin título')))
    
    if tags_index:
        for tag in sorted(tags_index.keys()):
            catalog += f"### #{tag}\n\n"
            for note_id, titulo in sorted(tags_index[tag]):
                catalog += f"- [[{note_id}]] - {titulo}\n"
            catalog += "\n"
    else:
        catalog += "*No hay tags aún.*\n\n"
    
    catalog += "---\n\n"
    
    # Índice por Estado
    catalog += "## 📊 Índice por Estado\n\n"
    
    for estado in ['completo', 'en_revision', 'borrador']:
        catalog += f"### {estado.replace('_', ' ').title()}\n\n"
        
        state_notes = [(nid, meta) for nid, meta in notes.items() 
                       if meta.get('estado') == estado]
        state_notes.sort(key=lambda x: x[0])
        
        if state_notes:
            for note_id, metadata in state_notes:
                titulo = metadata.get('titulo', 'Sin título')
                catalog += f"- [[{note_id}]] - {titulo}\n"
        else:
            catalog += f"*No hay notas en estado {estado} aún.*\n"
        
        catalog += "\n"
    
    catalog += "---\n\n"
    
    # Grafo de Conexiones
    catalog += "## 🔗 Grafo de Conexiones\n\n"
    
    catalog += "### Top 10 Nodos Más Conectados\n\n"
    
    if stats['most_connected']:
        for i, (note_id, count) in enumerate(stats['most_connected'][:10], 1):
            metadata = notes.get(note_id, {})
            titulo = metadata.get('titulo', 'Sin título')
            catalog += f"{i}. [[{note_id}]] - {titulo} ({count} conexiones)\n"
    else:
        catalog += "*No hay conexiones aún.*\n"
    
    catalog += "\n"
    
    catalog += "### Notas Huérfanas\n\n"
    
    if stats['orphan_notes']:
        catalog += "*Notas sin conexiones (considerar agregar enlaces):*\n\n"
        for note_id in sorted(stats['orphan_notes']):
            metadata = notes.get(note_id, {})
            titulo = metadata.get('titulo', 'Sin título')
            catalog += f"- [[{note_id}]] - {titulo}\n"
    else:
        catalog += "*No hay notas huérfanas. ¡Excelente conectividad!*\n"
    
    catalog += "\n---\n\n"
    
    # Métricas de Progreso
    catalog += "## 📈 Métricas de Progreso\n\n"
    
    catalog += "### Cobertura por Dominio\n\n"
    catalog += "| Dominio | Fenómenos | Objetivo Año 1 | Progreso |\n"
    catalog += "|---------|-----------|----------------|----------|\n"
    
    objetivos_dominio = {
        'biologia': 3,
        'economia': 2,
        'fisica': 3,
        'matematicas': 2,
        'sociologia': 2,
        'linguistica': 1,
        'informatica': 1,
        'neurociencia': 1,
    }
    
    total_actual = 0
    total_objetivo = 0
    
    for dominio, objetivo in objetivos_dominio.items():
        actual = stats['by_domain'].get(dominio, 0)
        total_actual += actual
        total_objetivo += objetivo
        progreso = (actual / objetivo * 100) if objetivo > 0 else 0
        catalog += f"| {dominio.replace('_', ' ').title()} | {actual} | {objetivo} | {progreso:.0f}% |\n"
    
    progreso_total = (total_actual / total_objetivo * 100) if total_objetivo > 0 else 0
    catalog += f"| **TOTAL** | **{total_actual}** | **{total_objetivo}** | **{progreso_total:.0f}%** |\n\n"
    
    catalog += "### Estado de Implementación\n\n"
    catalog += "| Estado | Fenómenos | Porcentaje |\n"
    catalog += "|--------|-----------|------------|\n"
    
    fenomenos_total = stats['by_type'].get('fenomeno', 0)
    
    for estado in ['completo', 'en_revision', 'borrador']:
        count = stats['by_state'].get(estado, 0)
        porcentaje = (count / fenomenos_total * 100) if fenomenos_total > 0 else 0
        catalog += f"| {estado.replace('_', ' ').title()} | {count} | {porcentaje:.0f}% |\n"
    
    catalog += f"| **TOTAL** | **{fenomenos_total}** | **100%** |\n\n"
    
    catalog += "---\n\n"
    
    # Herramientas
    catalog += """## 🛠️ Herramientas

### Scripts Disponibles

- `scripts/zettelkasten/update_catalog.py` - Regenerar este catálogo
- `scripts/zettelkasten/validate_zettelkasten.py` - Validar consistencia
- `scripts/zettelkasten/create_note.py` - Crear nueva nota
- `scripts/zettelkasten/visualize_graph.py` - Visualizar grafo de conexiones
- `scripts/zettelkasten/search_zettelkasten.py` - Búsqueda avanzada

### Comandos Útiles

```bash
# Regenerar catálogo
python scripts/zettelkasten/update_catalog.py

# Validar consistencia
python scripts/zettelkasten/validate_zettelkasten.py

# Crear nueva nota de fenómeno
python scripts/zettelkasten/create_note.py --type fenomeno --interactive

# Visualizar grafo
python scripts/zettelkasten/visualize_graph.py --output graph.html
```

---

**Este catálogo se regenera automáticamente. No editar manualmente.**
"""
    
    return catalog


def main():
    parser = argparse.ArgumentParser(
        description="Regenerar el catálogo maestro del Zettelkasten"
    )
    parser.add_argument(
        "--output",
        default=str(CATALOG_PATH),
        help="Ruta del archivo de salida"
    )
    
    args = parser.parse_args()
    
    print("🔍 Recolectando notas del Zettelkasten...")
    notes = collect_all_notes()
    print(f"   Encontradas {len(notes)} notas")
    
    print("📊 Generando estadísticas...")
    stats = generate_statistics(notes)
    print(f"   Total de conexiones: {stats['total_connections']}")
    print(f"   Densidad: {stats['density']:.2f} conexiones/nota")
    
    print("📝 Generando catálogo...")
    catalog_content = generate_catalog(notes, stats)
    
    print(f"💾 Guardando catálogo en {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(catalog_content)
    
    print("✅ Catálogo maestro actualizado exitosamente!")
    
    # Mostrar advertencias
    if stats['orphan_notes']:
        print(f"\n⚠️  Advertencia: {len(stats['orphan_notes'])} notas huérfanas detectadas")
        print("   Considerar agregar conexiones a estas notas")


if __name__ == "__main__":
    main()


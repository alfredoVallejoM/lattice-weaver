#!/usr/bin/env python3
"""
Script para búsqueda avanzada en el Zettelkasten.

Permite buscar notas por contenido, tags, metadatos y encontrar caminos entre notas.

Uso:
    python search_zettelkasten.py --query "juegos"
    python search_zettelkasten.py --tag equilibrio
    python search_zettelkasten.py --type fenomeno --domain biologia
    python search_zettelkasten.py --path F001 F010
"""

import os
import sys
import argparse
import re
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque

# Añadir el directorio raíz al path
SCRIPT_DIR = Path(__file__).parent
TRACK_DIR = SCRIPT_DIR.parent.parent
ZETTEL_DIR = TRACK_DIR / "zettelkasten"


def extract_metadata(filepath: Path) -> Dict:
    """Extrae los metadatos YAML de una nota."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        return {}
    
    yaml_content = match.group(1)
    try:
        metadata = yaml.safe_load(yaml_content)
        metadata['_filepath'] = filepath
        metadata['_content'] = content
        return metadata
    except yaml.YAMLError:
        return {}


def extract_links(filepath: Path) -> List[str]:
    """Extrae todos los enlaces [[ID]] de una nota."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    links = re.findall(r'\[\[([A-Z]\d{3})', content)
    return list(set(links))


def collect_all_notes() -> Dict[str, Dict]:
    """Recolecta todas las notas del Zettelkasten."""
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


def search_by_content(notes: Dict[str, Dict], query: str) -> List[Tuple[str, Dict, List[str]]]:
    """
    Busca notas por contenido.
    
    Args:
        notes: Diccionario de notas
        query: Texto a buscar
    
    Returns:
        Lista de tuplas (note_id, metadata, contextos)
    """
    results = []
    query_lower = query.lower()
    
    for note_id, metadata in notes.items():
        content = metadata.get('_content', '')
        content_lower = content.lower()
        
        if query_lower in content_lower:
            # Extraer contextos (líneas que contienen la query)
            contexts = []
            for line in content.split('\n'):
                if query_lower in line.lower():
                    contexts.append(line.strip())
            
            results.append((note_id, metadata, contexts))
    
    return results


def search_by_tag(notes: Dict[str, Dict], tag: str) -> List[Tuple[str, Dict]]:
    """
    Busca notas por tag.
    
    Args:
        notes: Diccionario de notas
        tag: Tag a buscar
    
    Returns:
        Lista de tuplas (note_id, metadata)
    """
    results = []
    tag_lower = tag.lower()
    
    for note_id, metadata in notes.items():
        tags = metadata.get('tags', [])
        if isinstance(tags, list):
            if any(t.lower() == tag_lower for t in tags):
                results.append((note_id, metadata))
    
    return results


def search_by_filters(notes: Dict[str, Dict], 
                      note_type: Optional[str] = None,
                      domain: Optional[str] = None,
                      category: Optional[str] = None,
                      state: Optional[str] = None) -> List[Tuple[str, Dict]]:
    """
    Busca notas aplicando filtros.
    
    Args:
        notes: Diccionario de notas
        note_type: Tipo de nota a buscar
        domain: Dominio a buscar
        category: Categoría a buscar
        state: Estado a buscar
    
    Returns:
        Lista de tuplas (note_id, metadata)
    """
    results = []
    
    for note_id, metadata in notes.items():
        # Filtrar por tipo
        if note_type and metadata.get('tipo') != note_type:
            continue
        
        # Filtrar por dominio
        if domain:
            dominios = metadata.get('dominios', [])
            if not isinstance(dominios, list) or domain not in dominios:
                continue
        
        # Filtrar por categoría
        if category:
            categorias = metadata.get('categorias', [])
            if not isinstance(categorias, list) or category not in categorias:
                continue
        
        # Filtrar por estado
        if state and metadata.get('estado') != state:
            continue
        
        results.append((note_id, metadata))
    
    return results


def find_path(notes: Dict[str, Dict], start_id: str, end_id: str) -> Optional[List[str]]:
    """
    Encuentra el camino más corto entre dos notas.
    
    Args:
        notes: Diccionario de notas
        start_id: ID de la nota inicial
        end_id: ID de la nota final
    
    Returns:
        Lista de IDs del camino, o None si no hay camino
    """
    if start_id not in notes or end_id not in notes:
        return None
    
    # BFS para encontrar el camino más corto
    queue = deque([(start_id, [start_id])])
    visited = {start_id}
    
    while queue:
        current_id, path = queue.popleft()
        
        if current_id == end_id:
            return path
        
        # Explorar vecinos
        links = notes[current_id].get('_links', [])
        for linked_id in links:
            if linked_id in notes and linked_id not in visited:
                visited.add(linked_id)
                queue.append((linked_id, path + [linked_id]))
    
    return None


def find_related_notes(notes: Dict[str, Dict], note_id: str, max_distance: int = 2) -> Dict[int, List[str]]:
    """
    Encuentra notas relacionadas a una distancia máxima.
    
    Args:
        notes: Diccionario de notas
        note_id: ID de la nota base
        max_distance: Distancia máxima
    
    Returns:
        Diccionario {distancia: [lista de IDs]}
    """
    if note_id not in notes:
        return {}
    
    related = defaultdict(list)
    queue = deque([(note_id, 0)])
    visited = {note_id}
    
    while queue:
        current_id, distance = queue.popleft()
        
        if distance > 0:
            related[distance].append(current_id)
        
        if distance < max_distance:
            links = notes[current_id].get('_links', [])
            for linked_id in links:
                if linked_id in notes and linked_id not in visited:
                    visited.add(linked_id)
                    queue.append((linked_id, distance + 1))
    
    return dict(related)


def print_search_results(results: List, result_type: str):
    """
    Imprime los resultados de búsqueda.
    
    Args:
        results: Lista de resultados
        result_type: Tipo de resultado ('content', 'tag', 'filter')
    """
    if not results:
        print("\n❌ No se encontraron resultados.\n")
        return
    
    print(f"\n✅ Se encontraron {len(results)} resultados:\n")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        if result_type == 'content':
            note_id, metadata, contexts = result
            titulo = metadata.get('titulo', 'Sin título')
            tipo = metadata.get('tipo', 'unknown')
            
            print(f"\n{i}. [[{note_id}]] - {titulo}")
            print(f"   Tipo: {tipo}")
            print(f"   Contextos encontrados:")
            for context in contexts[:3]:  # Mostrar máximo 3 contextos
                print(f"     • {context[:100]}...")
        else:
            note_id, metadata = result
            titulo = metadata.get('titulo', 'Sin título')
            tipo = metadata.get('tipo', 'unknown')
            estado = metadata.get('estado', 'unknown')
            
            print(f"\n{i}. [[{note_id}]] - {titulo}")
            print(f"   Tipo: {tipo} | Estado: {estado}")
            
            dominios = metadata.get('dominios', [])
            if isinstance(dominios, list) and dominios:
                print(f"   Dominios: {', '.join(dominios)}")
            
            tags = metadata.get('tags', [])
            if isinstance(tags, list) and tags:
                print(f"   Tags: {', '.join(tags)}")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Búsqueda avanzada en el Zettelkasten"
    )
    
    # Opciones de búsqueda
    parser.add_argument(
        "--query",
        help="Buscar por contenido"
    )
    parser.add_argument(
        "--tag",
        help="Buscar por tag"
    )
    parser.add_argument(
        "--type",
        choices=['fenomeno', 'categoria', 'isomorfismo', 'tecnica', 'dominio', 'concepto', 'mapeo'],
        help="Filtrar por tipo de nota"
    )
    parser.add_argument(
        "--domain",
        help="Filtrar por dominio"
    )
    parser.add_argument(
        "--category",
        help="Filtrar por categoría (ID, ej. C001)"
    )
    parser.add_argument(
        "--state",
        choices=['borrador', 'en_revision', 'completo'],
        help="Filtrar por estado"
    )
    parser.add_argument(
        "--path",
        nargs=2,
        metavar=('START_ID', 'END_ID'),
        help="Encontrar camino entre dos notas"
    )
    parser.add_argument(
        "--related",
        metavar='NOTE_ID',
        help="Encontrar notas relacionadas"
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=2,
        help="Distancia máxima para notas relacionadas (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Verificar que se proporcionó al menos una opción
    if not any([args.query, args.tag, args.type, args.domain, args.category, 
                args.state, args.path, args.related]):
        parser.print_help()
        sys.exit(1)
    
    print("🔍 Cargando Zettelkasten...")
    notes = collect_all_notes()
    print(f"   Cargadas {len(notes)} notas")
    
    # Búsqueda por contenido
    if args.query:
        print(f"\n🔎 Buscando por contenido: '{args.query}'")
        results = search_by_content(notes, args.query)
        print_search_results(results, 'content')
    
    # Búsqueda por tag
    if args.tag:
        print(f"\n🏷️  Buscando por tag: #{args.tag}")
        results = search_by_tag(notes, args.tag)
        print_search_results(results, 'tag')
    
    # Búsqueda por filtros
    if any([args.type, args.domain, args.category, args.state]):
        print("\n🔍 Aplicando filtros...")
        results = search_by_filters(notes, args.type, args.domain, args.category, args.state)
        print_search_results(results, 'filter')
    
    # Encontrar camino
    if args.path:
        start_id, end_id = args.path
        print(f"\n🛤️  Buscando camino de [[{start_id}]] a [[{end_id}]]...")
        
        path = find_path(notes, start_id, end_id)
        
        if path:
            print(f"\n✅ Camino encontrado (longitud: {len(path) - 1}):\n")
            for i, note_id in enumerate(path):
                metadata = notes[note_id]
                titulo = metadata.get('titulo', 'Sin título')
                
                if i == 0:
                    print(f"   🎯 [[{note_id}]] - {titulo}")
                elif i == len(path) - 1:
                    print(f"   {'   ' * i}└─> 🎯 [[{note_id}]] - {titulo}")
                else:
                    print(f"   {'   ' * i}└─> [[{note_id}]] - {titulo}")
        else:
            print(f"\n❌ No se encontró camino entre [[{start_id}]] y [[{end_id}]]")
    
    # Encontrar notas relacionadas
    if args.related:
        note_id = args.related
        print(f"\n🔗 Buscando notas relacionadas a [[{note_id}]] (distancia máx: {args.distance})...")
        
        if note_id not in notes:
            print(f"\n❌ Nota [[{note_id}]] no encontrada")
        else:
            related = find_related_notes(notes, note_id, args.distance)
            
            if related:
                print(f"\n✅ Notas relacionadas encontradas:\n")
                
                for distance in sorted(related.keys()):
                    note_ids = related[distance]
                    print(f"Distancia {distance}: {len(note_ids)} notas")
                    
                    for nid in note_ids[:10]:  # Mostrar máximo 10 por distancia
                        metadata = notes[nid]
                        titulo = metadata.get('titulo', 'Sin título')
                        print(f"  • [[{nid}]] - {titulo}")
                    
                    if len(note_ids) > 10:
                        print(f"  ... y {len(note_ids) - 10} más")
                    print()
            else:
                print(f"\n❌ No se encontraron notas relacionadas")


if __name__ == "__main__":
    main()


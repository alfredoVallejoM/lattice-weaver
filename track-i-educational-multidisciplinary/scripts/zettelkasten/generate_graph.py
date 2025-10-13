import json
import re
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Añadir el directorio raíz al path
SCRIPT_DIR = Path(__file__).parent
TRACK_DIR = SCRIPT_DIR.parent.parent
ZETTEL_DIR = TRACK_DIR / "zettelkasten"
TEMPLATE_DIR = TRACK_DIR / "templates"

# Mapeo de tipos a prefijos y directorios
TYPE_CONFIG = {
    "fenomeno": {"prefix": "F", "dir": "fenomenos"},
    "categoria": {"prefix": "C", "dir": "categorias"},
    "isomorfismo": {"prefix": "I", "dir": "isomorfismos"},
    "tecnica": {"prefix": "T", "dir": "tecnicas"},
    "dominio": {"prefix": "D", "dir": "dominios"},
    "concepto": {"prefix": "K", "dir": "conceptos"},
    "mapeo": {"prefix": "M", "dir": "mapeos"},
}

def get_note_id_from_filename(filename):
    # Intenta coincidir con el patrón ID_nombre.md
    match = re.match(r'([FCIKDTM]\d{3})_.*\.md', filename)
    if match:
        return match.group(1)
    # Si no coincide, intenta coincidir con el patrón ID.md
    match = re.match(r'([FCIKDTM]\d{3})\.md', filename)
    if match:
        return match.group(1)
    return None

def extract_metadata(filepath: Path) -> Dict:
    """
    Extrae los metadatos YAML de una nota.
    """
    metadata = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return metadata
    
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if match:
        yaml_content = match.group(1)
        try:
            yaml_data = yaml.safe_load(yaml_content)
            if yaml_data:
                metadata.update(yaml_data)
        except yaml.YAMLError:
            pass
    return metadata

def extract_links(filepath: Path) -> List[str]:
    """
    Extrae todos los enlaces [[ID]] o [[ID|texto]] de una nota.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    links = re.findall(r'\[\[([A-Z]\d{3})(?:\|.*?)?\]\]', content)
    return links

def collect_all_notes() -> Dict[str, Dict]:
    """
    Recolecta todas las notas del Zettelkasten.
    """
    notes = {}
    for subdir_config in TYPE_CONFIG.values():
        subdir = ZETTEL_DIR / subdir_config["dir"]
        if not subdir.is_dir():
            continue
        
        for filepath in subdir.glob("*.md"):
            metadata = extract_metadata(filepath)
            note_id_from_filename = get_note_id_from_filename(filepath.name)

            if 'id' not in metadata and note_id_from_filename:
                metadata['id'] = note_id_from_filename
            
            if 'titulo' not in metadata and note_id_from_filename:
                metadata['titulo'] = filepath.stem.replace(f'{note_id_from_filename}_', '').replace('_', ' ').title()

            note_id = metadata.get('id')
            if note_id:
                metadata['_filepath'] = filepath
                metadata['_links'] = extract_links(filepath)
                notes[note_id] = metadata
    return notes

def generate_graph_data(notes: Dict[str, Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Genera los datos de nodos y aristas para vis.js.
    """
    nodes = []
    edges = []
    
    for note_id, metadata in notes.items():
        node_label = f"{note_id}\n{metadata.get('titulo', 'Sin título')}"
        node_group = metadata.get('tipo', 'desconocido')
        
        nodes.append({
            'id': note_id,
            'label': node_label,
            'group': node_group,
            'title': f"Tipo: {node_group.capitalize()}\nTítulo: {metadata.get('titulo', 'Sin título')}\nArchivo: {metadata['_filepath'].name}"
        })
        
        for linked_id in metadata.get('_links', []):
            if linked_id in notes: # Solo añadir aristas a nodos existentes
                edges.append({
                    'from': note_id,
                    'to': linked_id,
                    'arrows': 'to'
                })
    
    return nodes, edges

def create_html_graph(nodes: List[Dict], edges: List[Dict], output_path: Path):
    """
    Crea un archivo HTML con la visualización del grafo usando vis.js.
    """
    nodes_json = json.dumps(nodes, indent=2)
    edges_json = json.dumps(edges, indent=2)

    template_path = TEMPLATE_DIR / "graph_template.html"
    with open(template_path, 'r', encoding='utf-8') as f:
        html_template = f.read()

    html_content = html_template.format(nodes_json=nodes_json, edges_json=edges_json)
    output_path.write_text(html_content, encoding='utf-8')
    print(f"Grafo generado exitosamente en {output_path}")

def main():
    print("🔍 Recolectando notas del Zettelkasten para generar el grafo...")
    notes = collect_all_notes()
    print(f"   Encontradas {len(notes)} notas.")
    
    nodes, edges = generate_graph_data(notes)
    print(f"   Generados {len(nodes)} nodos y {len(edges)} aristas.")
    
    output_file = TRACK_DIR / "zettelkasten_graph.html"
    create_html_graph(nodes, edges, output_file)

if __name__ == "__main__":
    main()


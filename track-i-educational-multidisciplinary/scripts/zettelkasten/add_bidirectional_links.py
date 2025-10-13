
import os
import re
import yaml
from pathlib import Path
from typing import List, Dict, Optional

# Añadir el directorio raíz al path
SCRIPT_DIR = Path(__file__).parent
TRACK_DIR = SCRIPT_DIR.parent.parent
ZETTEL_DIR = TRACK_DIR / "zettelkasten"

# Mapeo de tipos a prefijos y directorios
TYPE_CONFIG = {
    "fenomeno": {"prefix": "F", "dir": "fenomenos"},
    "categoria": {"prefix": "C", "dir": "categorias"},
    "isomorfismo": {"prefix": "I", "dir": "isomorfismos"},
    "tecnica": {"prefix": "T", "dir": "tecnicas"},
    "dominio": {"prefix": "D", "dir": "dominios"}, # Añadido el tipo dominio
    "concepto": {"prefix": "K", "dir": "conceptos"},
    "mapeo": {"prefix": "M", "dir": "mapeos"},
}

def get_note_id_from_filename(filename):
    match = re.match(r'([FCIKDT]\d{3})_.*\.md', filename)
    if match:
        return match.group(1)
    return None

def get_filename_from_note_id(note_id: str) -> Optional[Path]:
    prefix = note_id[0]
    for config_type, config_data in TYPE_CONFIG.items():
        if config_data["prefix"] == prefix:
            notes_dir = ZETTEL_DIR / config_data["dir"]
            if notes_dir.exists():
                for file in notes_dir.glob(f"{note_id}_*.md"):
                    return file
    return None

def extract_links(content: str) -> List[str]:
    return re.findall(r'\[\[([FCIKDT]\d{3})\]\]', content)

def get_note_type_from_id(note_id: str) -> str:
    prefix = note_id[0]
    for note_type, config_data in TYPE_CONFIG.items():
        if config_data["prefix"] == prefix:
            return note_type.capitalize()
    return "Desconocido"

def add_link_to_note(filepath: Path, target_id: str, source_id: str) -> bool:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if the link already exists
    if re.search(rf'\- \[\[{source_id}\]\]', content):
        return False # Link already exists

    source_type = get_note_type_from_id(source_id)
    new_link_line = f'- [[{source_id}]] - Conexión inversa con {source_type}.'

    # Find the 'Conexiones' section
    connections_section_match = re.search(r'\n## Conexiones\n(.*?)(?=\n## |\Z)', content, re.DOTALL)

    if connections_section_match:
        # Insert at the end of the 'Conexiones' section
        connections_content = connections_section_match.group(1)
        # Ensure there's a newline before adding the new link if the section isn't empty
        if connections_content.strip():
            new_connections_content = connections_content.rstrip() + '\n' + new_link_line
        else:
            new_connections_content = new_link_line
        content = content.replace(connections_section_match.group(0), f'\n## Conexiones\n{new_connections_content.strip()}\n')
    else:
        # If 'Conexiones' section doesn't exist, add it at the end of the file
        content += f'\n## Conexiones\n\n{new_link_line}\n'

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return True

def main():
    print("Iniciando el proceso de adición de enlaces bidireccionales...")
    all_notes = []
    for config_type, config_data in TYPE_CONFIG.items():
        notes_dir = ZETTEL_DIR / config_data["dir"]
        if notes_dir.exists():
            for file in notes_dir.glob("*.md"):
                if not file.name.startswith("template_"):
                    all_notes.append(file)

    # Extract all links first
    note_links = {}
    for note_path in all_notes:
        with open(note_path, 'r', encoding='utf-8') as f:
            content = f.read()
        note_id = get_note_id_from_filename(note_path.name)
        if note_id:
            note_links[note_id] = extract_links(content)

    # Add inverse links
    links_added_count = 0
    for source_id, targets in note_links.items():
        for target_id in targets:
            target_filepath = get_filename_from_note_id(target_id)
            if target_filepath:
                if add_link_to_note(target_filepath, target_id, source_id):
                    links_added_count += 1
            else:
                print(f"Advertencia: No se encontró el archivo para el ID de destino {target_id} (enlace desde {source_id})")
    
    print(f"Proceso completado. Se añadieron {links_added_count} enlaces bidireccionales.")

if __name__ == '__main__':
    main()


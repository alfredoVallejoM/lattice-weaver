import os
import sys
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Añadir el directorio raíz al path
SCRIPT_DIR = Path(__file__).parent
TRACK_DIR = SCRIPT_DIR.parent.parent
ZETTEL_DIR = TRACK_DIR / "zettelkasten"
TEMPLATES_DIR = TRACK_DIR / "templates"

# Mapeo de tipos a prefijos y directorios
TYPE_CONFIG = {
    "fenomeno": {"prefix": "F", "dir": "fenomenos", "template": "template_fenomeno.md"},
    "categoria": {"prefix": "C", "dir": "categorias", "template": "template_categoria.md"},
    "isomorfismo": {"prefix": "I", "dir": "isomorfismos", "template": "template_isomorfismo.md"},
    "tecnica": {"prefix": "T", "dir": "tecnicas", "template": "template_tecnica.md"},
    "dominio": {"prefix": "D", "dir": "dominios", "template": "template_dominio.md"},
    "concepto": {"prefix": "K", "dir": "conceptos", "template": "template_concepto.md"},
    "mapeo": {"prefix": "M", "dir": "mapeos", "template": None},
}


def get_next_id(note_type: str) -> str:
    """
    Obtiene el siguiente ID disponible para un tipo de nota.
    
    Args:
        note_type: Tipo de nota (fenomeno, categoria, etc.)
    
    Returns:
        ID en formato PREFIX### (ej. F001, C003)
    """
    config = TYPE_CONFIG[note_type]
    prefix = config["prefix"]
    notes_dir = ZETTEL_DIR / config["dir"]
    
    # Buscar todos los archivos con el prefijo
    existing_ids = []
    if notes_dir.exists():
        for file in notes_dir.glob(f"{prefix}*.md"):
            match = re.match(rf"{prefix}(\d+)_", file.name)
            if match:
                existing_ids.append(int(match.group(1)))
    
    # Obtener el siguiente número
    next_num = max(existing_ids, default=0) + 1
    return f"{prefix}{next_num:03d}"


def load_template(note_type: str) -> Optional[str]:
    """
    Carga el template para un tipo de nota.
    
    Args:
        note_type: Tipo de nota
    
    Returns:
        Contenido del template o None si no existe
    """
    config = TYPE_CONFIG[note_type]
    template_name = config.get("template")
    
    if not template_name:
        return None
    
    template_path = TEMPLATES_DIR / template_name
    if not template_path.exists():
        return None
    
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def create_note_interactive(note_type: str) -> Dict[str, str]:
    """
    Crea una nota de forma interactiva pidiendo información al usuario.
    
    Args:
        note_type: Tipo de nota a crear
    
    Returns:
        Diccionario con la información de la nota
    """
    print(f"\n=== Creando nueva nota de tipo: {note_type.upper()} ===\n")
    
    # Obtener ID automáticamente
    note_id = input(f"ID de la nota (dejar en blanco para {get_next_id(note_type)}): ").strip()
    if not note_id:
        note_id = get_next_id(note_type)
    print(f"ID asignado: {note_id}")
    
    # Pedir información básica
    titulo = input("\nTítulo de la nota: ").strip()
    
    # Información específica por tipo
    info = {
        "id": note_id,
        "tipo": note_type,
        "titulo": titulo,
        "fecha_creacion": datetime.now().strftime("%Y-%m-%d"),
        "fecha_modificacion": datetime.now().strftime("%Y-%m-%d"),
        "estado": "borrador",
    }
    
    if note_type == "fenomeno":
        dominios_str = input("Dominios (separados por coma): ").strip()
        info["dominios"] = [d.strip() for d in dominios_str.split(",") if d.strip()]
        
        categorias_str = input("Categorías (IDs separados por coma, ej. C001,C002): ").strip()
        info["categorias"] = [c.strip() for c in categorias_str.split(",") if c.strip()]
        
        tags_str = input("Tags (separados por coma): ").strip()
        info["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]
        
        prioridad = input("Prioridad (maxima/alta/media/baja) [media]: ").strip() or "media"
        info["prioridad"] = prioridad
    
    elif note_type == "isomorfismo":
        nivel = input("Nivel (exacto/fuerte/analogia) [fuerte]: ").strip() or "fuerte"
        info["nivel"] = nivel
        
        fenomenos_str = input("Fenómenos relacionados (IDs separados por coma, ej. F001,F002): ").strip()
        info["fenomenos"] = [f.strip() for f in fenomenos_str.split(",") if f.strip()]
        
        dominios_str = input("Dominios (separados por coma): ").strip()
        info["dominios"] = [d.strip() for d in dominios_str.split(",") if d.strip()]
        
        categorias_str = input("Categorías (IDs separados por coma): ").strip()
        info["categorias"] = [c.strip() for c in categorias_str.split(",") if c.strip()]
        
        tags_str = input("Tags (separados por coma): ").strip()
        info["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]
        
        info["validacion"] = "pendiente"
    
    elif note_type == "categoria":
        tags_str = input("Tags (separados por coma): ").strip()
        info["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]
        
        info["fenomenos_count"] = 0
        info["dominios_count"] = 0
    
    elif note_type == "tecnica":
        dominio_origen = input("Dominio de origen: ").strip()
        info["dominio_origen"] = dominio_origen
        
        categorias_str = input("Categorías aplicables (IDs separados por coma): ").strip()
        info["categorias_aplicables"] = [c.strip() for c in categorias_str.split(",") if c.strip()]
        
        tags_str = input("Tags (separados por coma): ").strip()
        info["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]
        
        info["implementado"] = False

    elif note_type == "dominio":
        tags_str = input("Tags (separados por coma): ").strip()
        info["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]
        info["dominio_origen"] = info["titulo"]
        info["categorias_aplicables"] = []
        info["prioridad"] = input("Prioridad (maxima/alta/media/baja) [media]: ").strip() or "media"

    elif note_type == "concepto":
        dominios_str = input("Dominios (separados por coma): ").strip()
        info["dominios"] = [d.strip() for d in dominios_str.split(",") if d.strip()]
        
        tags_str = input("Tags (separados por coma): ").strip()
        info["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]
        
        prioridad = input("Prioridad (maxima/alta/media/baja) [media]: ").strip() or "media"
        info["prioridad"] = prioridad
    
    return info


def fill_template(template: str, info: Dict[str, str]) -> str:
    """
    Rellena un template con la información proporcionada.
    
    Args:
        template: Contenido del template
        info: Diccionario con información de la nota
    
    Returns:
        Template rellenado
    """
    # Reemplazar placeholders genéricos en el front matter y el cuerpo
    for key, value in info.items():
        if isinstance(value, list):
            # Format lists as YAML lists for front matter
            template = template.replace(f"{{{{{key}}}}}", str(value))
        else:
            template = template.replace(f"{{{{{key}}}}}", str(value))

    # Replace specific placeholders that might not be in info dict directly
    template = template.replace("{{date}}", info["fecha_creacion"])
    
    # Remove any remaining {{...}} placeholders that were not filled
    template = re.sub(r"{{\w+}}", "", template)

    return template


def save_note(note_type: str, info: Dict[str, str], content: str) -> Path:
    """
    Guarda la nota en el directorio apropiado.
    
    Args:
        note_type: Tipo de nota
        info: Información de la nota
        content: Contenido de la nota
    
    Returns:
        Path al archivo creado
    """
    config = TYPE_CONFIG[note_type]
    notes_dir = ZETTEL_DIR / config["dir"]
    notes_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear nombre de archivo
    titulo_slug = re.sub(r'[^a-z0-9]+', '_', info['titulo'].lower()).strip('_')
    filename = f"{info['id']}_{titulo_slug}.md"
    filepath = notes_dir / filename
    
    # Guardar archivo
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Crear nueva nota en el Zettelkasten del Track I"
    )
    parser.add_argument(
        "--type",
        choices=list(TYPE_CONFIG.keys()),
        required=True,
        help="Tipo de nota a crear"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Modo interactivo (pide información al usuario)"
    )
    parser.add_argument(
        "--title",
        help="Título de la nota (modo no interactivo)"
    )
    parser.add_argument(
        "--id",
        help="ID de la nota (opcional, se genera automáticamente si no se provee)"
    )
    parser.add_argument(
        "--dominios",
        help="Dominios de la nota (separados por coma, modo no interactivo)"
    )
    parser.add_argument(
        "--categorias",
        help="Categorías de la nota (IDs separados por coma, modo no interactivo)"
    )
    parser.add_argument(
        "--tags",
        help="Tags de la nota (separados por coma, modo no interactivo)"
    )
    parser.add_argument(
        "--prioridad",
        help="Prioridad de la nota (maxima/alta/media/baja, modo no interactivo)"
    )
    parser.add_argument(
        "--nivel",
        help="Nivel del isomorfismo (exacto/fuerte/analogia, modo no interactivo)"
    )
    parser.add_argument(
        "--fenomenos",
        help="Fenómenos relacionados (IDs separados por coma, modo no interactivo)"
    )
    parser.add_argument(
        "--dominio_origen",
        help="Dominio de origen de la técnica (modo no interactivo)"
    )
    parser.add_argument(
        "--categorias_aplicables",
        help="Categorías aplicables a la técnica (IDs separados por coma, modo no interactivo)"
    )
    
    args = parser.parse_args()
    
    # Verificar que los directorios existen
    if not ZETTEL_DIR.exists():
        print(f"Error: Directorio zettelkasten no encontrado en {ZETTEL_DIR}")
        sys.exit(1)
    
    if not TEMPLATES_DIR.exists():
        print(f"Error: Directorio templates no encontrado en {TEMPLATES_DIR}")
        sys.exit(1)
    
    info = {}
    # Modo interactivo
    if args.interactive:
        info = create_note_interactive(args.type)
    else:
        # Modo no interactivo
        if not args.title:
            print("Error: --title es requerido en modo no interactivo")
            sys.exit(1)
        
        note_id = args.id if args.id else get_next_id(args.type)
        info = {
            "id": note_id,
            "tipo": args.type,
            "titulo": args.title,
            "fecha_creacion": datetime.now().strftime("%Y-%m-%d"),
            "fecha_modificacion": datetime.now().strftime("%Y-%m-%d"),
            "estado": "borrador",
        }

        if args.dominios:
            info["dominios"] = [d.strip() for d in args.dominios.split(",") if d.strip()]
        if args.categorias:
            info["categorias"] = [c.strip() for c in args.categorias.split(",") if c.strip()]
        if args.tags:
            info["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
        if args.prioridad:
            info["prioridad"] = args.prioridad
        if args.nivel:
            info["nivel"] = args.nivel
        if args.fenomenos:
            info["fenomenos"] = [f.strip() for f in args.fenomenos.split(",") if f.strip()]
        if args.dominio_origen:
            info["dominio_origen"] = args.dominio_origen
        if args.categorias_aplicables:
            info["categorias_aplicables"] = [c.strip() for c in args.categorias_aplicables.split(",") if c.strip()]

        # Default values for specific types if not provided via args
        if args.type == "isomorfismo" and "nivel" not in info:
            info["nivel"] = "fuerte"
        if args.type == "isomorfismo" and "validacion" not in info:
            info["validacion"] = "pendiente"
        if args.type == "tecnica" and "implementado" not in info:
            info["implementado"] = False
        if args.type == "categoria" and "fenomenos_count" not in info:
            info["fenomenos_count"] = 0
        if args.type == "categoria" and "dominios_count" not in info:
            info["dominios_count"] = 0
        if args.type == "dominio" and "dominio_origen" not in info:
            info["dominio_origen"] = info["titulo"]
        if args.type == "dominio" and "categorias_aplicables" not in info:
            info["categorias_aplicables"] = []
        if args.type == "dominio" and "prioridad" not in info:
            info["prioridad"] = "media"
        if args.type == "concepto" and "prioridad" not in info:
            info["prioridad"] = "media"

    template_content = load_template(args.type)
    if template_content:
        final_content = fill_template(template_content, info)
    else:
        # Si no hay template, crear un front matter básico y el título
        front_matter = yaml.dump(info, allow_unicode=True, sort_keys=False)
        final_content = f"---\n{front_matter}---\n\n# {info['titulo']}\n\n"

    filepath = save_note(args.type, info, final_content)
    print(f"Nota creada exitosamente: {filepath}")

if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Script para visualizar el grafo de conexiones del Zettelkasten.

Genera visualizaciones interactivas del grafo de notas y sus conexiones.

Uso:
    python visualize_graph.py --output graph.html
    python visualize_graph.py --format graphml --output graph.graphml
"""

import os
import sys
import argparse
import re
import yaml
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Añadir el directorio raíz al path
SCRIPT_DIR = Path(__file__).parent
TRACK_DIR = SCRIPT_DIR.parent.parent
ZETTEL_DIR = TRACK_DIR / "zettelkasten"


def extract_metadata(filepath: Path) -> Dict:
    """
    Extrae los metadatos YAML de una nota.
    
    Args:
        filepath: Path al archivo
    
    Returns:
        Diccionario con los metadatos
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        return {}
    
    yaml_content = match.group(1)
    try:
        metadata = yaml.safe_load(yaml_content)
        metadata['_filepath'] = str(filepath.relative_to(TRACK_DIR))
        return metadata
    except yaml.YAMLError:
        return {}


def extract_links(filepath: Path) -> List[str]:
    """
    Extrae todos los enlaces [[ID]] de una nota.
    
    Args:
        filepath: Path al archivo
    
    Returns:
        Lista de IDs enlazados
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    links = re.findall(r'\[\[([A-Z]\d{3})', content)
    return list(set(links))


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


def generate_html_visualization(notes: Dict[str, Dict], output_path: str):
    """
    Genera una visualización HTML interactiva usando vis.js.
    
    Args:
        notes: Diccionario de notas
        output_path: Path del archivo de salida
    """
    # Preparar datos para vis.js
    nodes_data = []
    edges_data = []
    
    # Mapeo de tipos a colores
    type_colors = {
        'fenomeno': '#4CAF50',
        'categoria': '#2196F3',
        'isomorfismo': '#FF9800',
        'tecnica': '#9C27B0',
        'dominio': '#F44336',
        'concepto': '#00BCD4',
        'mapeo': '#795548',
    }
    
    # Crear nodos
    for note_id, metadata in notes.items():
        note_type = metadata.get('tipo', 'unknown')
        titulo = metadata.get('titulo', 'Sin título')
        estado = metadata.get('estado', 'unknown')
        
        # Tamaño basado en número de conexiones
        num_connections = len(metadata.get('_links', []))
        size = 10 + num_connections * 2
        
        # Color basado en tipo
        color = type_colors.get(note_type, '#999999')
        
        # Forma basada en estado
        shape = 'dot'
        if estado == 'completo':
            shape = 'diamond'
        elif estado == 'en_revision':
            shape = 'square'
        
        nodes_data.append({
            'id': note_id,
            'label': f"{note_id}\n{titulo[:30]}",
            'title': f"{note_id}: {titulo}\nTipo: {note_type}\nEstado: {estado}\nConexiones: {num_connections}",
            'color': color,
            'size': size,
            'shape': shape,
        })
    
    # Crear aristas
    edge_id = 0
    added_edges = set()
    
    for note_id, metadata in notes.items():
        links = metadata.get('_links', [])
        for linked_id in links:
            if linked_id in notes:
                # Evitar duplicados (A->B y B->A)
                edge_key = tuple(sorted([note_id, linked_id]))
                if edge_key not in added_edges:
                    edges_data.append({
                        'id': edge_id,
                        'from': note_id,
                        'to': linked_id,
                        'arrows': 'to',
                    })
                    edge_id += 1
                    added_edges.add(edge_key)
    
    # Generar HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Grafo del Zettelkasten - Track I</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }}
        #header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        #controls {{
            background-color: #ecf0f1;
            padding: 15px;
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        #mynetwork {{
            width: 100%;
            height: 800px;
            border: 1px solid lightgray;
        }}
        #info {{
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
        }}
        button {{
            padding: 10px 20px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }}
        button:hover {{
            background-color: #2980b9;
        }}
        #stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>Grafo del Zettelkasten - Track I</h1>
        <p>Visualización interactiva de notas y conexiones</p>
    </div>
    
    <div id="controls">
        <button onclick="network.fit()">Ajustar Vista</button>
        <button onclick="togglePhysics()">Toggle Física</button>
        <button onclick="exportImage()">Exportar Imagen</button>
    </div>
    
    <div id="mynetwork"></div>
    
    <div id="info">
        <h2>Estadísticas</h2>
        <div id="stats">
            <div class="stat-card">
                <div class="stat-value">{len(notes)}</div>
                <div class="stat-label">Total de Notas</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(edges_data)}</div>
                <div class="stat-label">Conexiones</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(edges_data)/max(len(notes), 1):.2f}</div>
                <div class="stat-label">Densidad</div>
            </div>
        </div>
        
        <h3>Leyenda</h3>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: {type_colors['fenomeno']}"></div>
                <span>Fenómeno</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {type_colors['categoria']}"></div>
                <span>Categoría</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {type_colors['isomorfismo']}"></div>
                <span>Isomorfismo</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {type_colors['tecnica']}"></div>
                <span>Técnica</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {type_colors['concepto']}"></div>
                <span>Concepto</span>
            </div>
        </div>
        
        <h3>Instrucciones</h3>
        <ul>
            <li>Haz clic y arrastra para mover el grafo</li>
            <li>Usa la rueda del ratón para hacer zoom</li>
            <li>Haz clic en un nodo para ver sus detalles</li>
            <li>Haz doble clic en un nodo para centrarlo</li>
        </ul>
    </div>
    
    <script type="text/javascript">
        // Datos del grafo
        var nodes = new vis.DataSet({json.dumps(nodes_data)});
        var edges = new vis.DataSet({json.dumps(edges_data)});
        
        // Crear red
        var container = document.getElementById('mynetwork');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        var options = {{
            nodes: {{
                font: {{
                    size: 12,
                    face: 'Arial'
                }},
                borderWidth: 2,
                borderWidthSelected: 4
            }},
            edges: {{
                width: 2,
                color: {{
                    color: '#848484',
                    highlight: '#3498db',
                    hover: '#3498db'
                }},
                smooth: {{
                    type: 'continuous'
                }}
            }},
            physics: {{
                stabilization: false,
                barnesHut: {{
                    gravitationalConstant: -8000,
                    springConstant: 0.04,
                    springLength: 95
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                navigationButtons: true,
                keyboard: true
            }}
        }};
        
        var network = new vis.Network(container, data, options);
        
        // Event listeners
        network.on("click", function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                console.log("Nodo seleccionado:", node);
            }}
        }});
        
        // Funciones de control
        var physicsEnabled = true;
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{physics: {{enabled: physicsEnabled}}}});
        }}
        
        function exportImage() {{
            var canvas = container.getElementsByTagName('canvas')[0];
            var dataURL = canvas.toDataURL('image/png');
            var link = document.createElement('a');
            link.download = 'zettelkasten_graph.png';
            link.href = dataURL;
            link.click();
        }}
    </script>
</body>
</html>"""
    
    # Guardar archivo
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def generate_graphml(notes: Dict[str, Dict], output_path: str):
    """
    Genera un archivo GraphML para importar en otras herramientas.
    
    Args:
        notes: Diccionario de notas
        output_path: Path del archivo de salida
    """
    graphml = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="d0" for="node" attr.name="titulo" attr.type="string"/>
  <key id="d1" for="node" attr.name="tipo" attr.type="string"/>
  <key id="d2" for="node" attr.name="estado" attr.type="string"/>
  <graph id="G" edgedefault="directed">
"""
    
    # Añadir nodos
    for note_id, metadata in notes.items():
        titulo = metadata.get('titulo', 'Sin título').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        tipo = metadata.get('tipo', 'unknown')
        estado = metadata.get('estado', 'unknown')
        
        graphml += f"""    <node id="{note_id}">
      <data key="d0">{titulo}</data>
      <data key="d1">{tipo}</data>
      <data key="d2">{estado}</data>
    </node>
"""
    
    # Añadir aristas
    edge_id = 0
    added_edges = set()
    
    for note_id, metadata in notes.items():
        links = metadata.get('_links', [])
        for linked_id in links:
            if linked_id in notes:
                edge_key = tuple(sorted([note_id, linked_id]))
                if edge_key not in added_edges:
                    graphml += f"""    <edge id="e{edge_id}" source="{note_id}" target="{linked_id}"/>
"""
                    edge_id += 1
                    added_edges.add(edge_key)
    
    graphml += """  </graph>
</graphml>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(graphml)


def main():
    parser = argparse.ArgumentParser(
        description="Visualizar el grafo de conexiones del Zettelkasten"
    )
    parser.add_argument(
        "--output",
        default="zettelkasten_graph.html",
        help="Archivo de salida"
    )
    parser.add_argument(
        "--format",
        choices=['html', 'graphml'],
        default='html',
        help="Formato de salida"
    )
    
    args = parser.parse_args()
    
    print("🔍 Recolectando notas del Zettelkasten...")
    notes = collect_all_notes()
    print(f"   Encontradas {len(notes)} notas")
    
    # Calcular estadísticas
    total_connections = sum(len(meta.get('_links', [])) for meta in notes.values())
    print(f"   Total de conexiones: {total_connections}")
    
    print(f"📊 Generando visualización en formato {args.format}...")
    
    if args.format == 'html':
        generate_html_visualization(notes, args.output)
    elif args.format == 'graphml':
        generate_graphml(notes, args.output)
    
    print(f"✅ Visualización guardada en: {args.output}")
    
    if args.format == 'html':
        print(f"\n💡 Abre el archivo en tu navegador para ver el grafo interactivo:")
        print(f"   file://{Path(args.output).absolute()}")


if __name__ == "__main__":
    main()


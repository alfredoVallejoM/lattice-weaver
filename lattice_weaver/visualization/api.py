"""
API REST para el SearchSpaceVisualizer.

Esta API permite a aplicaciones web (Track E) interactuar con el visualizador
para generar visualizaciones y reportes de forma programática.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

from typing import Optional, List
from pathlib import Path
import json
import tempfile

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from .search_viz import (
    load_trace,
    plot_search_tree,
    plot_domain_evolution,
    plot_backtrack_heatmap,
    plot_timeline,
    plot_variable_statistics,
    compare_traces,
    generate_report,
    generate_advanced_report
)


# Crear aplicación Flask
import logging

app = Flask(__name__)
CORS(app)  # Habilitar CORS para requests desde el frontend

# Configurar logging para la aplicación Flask
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
handler.setFormatter(formatter)
app.logger.addHandler(handler)



@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint de health check.
    
    Returns:
        JSON con el estado del servicio
    """
    return jsonify({
        'status': 'healthy',
        'service': 'LatticeWeaver Visualizer API',
        'version': '1.0'
    })


@app.route('/api/v1/visualize/tree', methods=['POST'])
def visualize_tree():
    """
    Genera visualización del árbol de búsqueda.
    
    Request Body:
        {
            "trace_path": str,  # Ruta del archivo de trace
            "max_nodes": int    # Opcional, máximo de nodos a visualizar
        }
    
    Returns:
        JSON con la figura de Plotly
    """
    try:
        data = request.get_json()
        trace_path = data.get('trace_path')
        max_nodes = data.get('max_nodes', 1000)
        
        if not trace_path:
            return jsonify({'error': 'trace_path is required'}), 400
        
        # Cargar trace
        df = load_trace(trace_path)
        
        # Generar visualización
        fig = plot_search_tree(df, max_nodes=max_nodes)
        
        return jsonify({
            'success': True,
            'figure': json.loads(fig.to_json())
        })
    
    except FileNotFoundError:
        return jsonify({'error': 'Trace file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/visualize/domain', methods=['POST'])
def visualize_domain():
    """
    Genera visualización de evolución de dominios.
    
    Request Body:
        {
            "trace_path": str
        }
    
    Returns:
        JSON con la figura de Plotly
    """
    try:
        data = request.get_json()
        trace_path = data.get('trace_path')
        
        if not trace_path:
            return jsonify({'error': 'trace_path is required'}), 400
        
        df = load_trace(trace_path)
        fig = plot_domain_evolution(df)
        
        return jsonify({
            'success': True,
            'figure': json.loads(fig.to_json())
        })
    
    except FileNotFoundError:
        return jsonify({'error': 'Trace file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/visualize/heatmap', methods=['POST'])
def visualize_heatmap():
    """
    Genera heatmap de backtracks.
    
    Request Body:
        {
            "trace_path": str
        }
    
    Returns:
        JSON con la figura de Plotly
    """
    try:
        data = request.get_json()
        trace_path = data.get('trace_path')
        
        if not trace_path:
            return jsonify({'error': 'trace_path is required'}), 400
        
        df = load_trace(trace_path)
        fig = plot_backtrack_heatmap(df)
        
        return jsonify({
            'success': True,
            'figure': json.loads(fig.to_json())
        })
    
    except FileNotFoundError:
        return jsonify({'error': 'Trace file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/visualize/timeline', methods=['POST'])
def visualize_timeline():
    """
    Genera visualización de línea de tiempo.
    
    Request Body:
        {
            "trace_path": str
        }
    
    Returns:
        JSON con la figura de Plotly
    """
    try:
        data = request.get_json()
        trace_path = data.get('trace_path')
        
        if not trace_path:
            return jsonify({'error': 'trace_path is required'}), 400
        
        df = load_trace(trace_path)
        fig = plot_timeline(df)
        
        return jsonify({
            'success': True,
            'figure': json.loads(fig.to_json())
        })
    
    except FileNotFoundError:
        return jsonify({'error': 'Trace file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/visualize/variable-stats', methods=['POST'])
def visualize_variable_stats():
    """
    Genera estadísticas por variable.
    
    Request Body:
        {
            "trace_path": str
        }
    
    Returns:
        JSON con la figura de Plotly
    """
    try:
        data = request.get_json()
        trace_path = data.get('trace_path')
        
        if not trace_path:
            return jsonify({'error': 'trace_path is required'}), 400
        
        df = load_trace(trace_path)
        fig = plot_variable_statistics(df)
        
        return jsonify({
            'success': True,
            'figure': json.loads(fig.to_json())
        })
    
    except FileNotFoundError:
        return jsonify({'error': 'Trace file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/compare', methods=['POST'])
def compare():
    """
    Compara múltiples traces.
    
    Request Body:
        {
            "traces": {
                "name1": "path1",
                "name2": "path2",
                ...
            },
            "metric": str  # Opcional: 'nodes_explored', 'backtracks', 'time', 'backtrack_rate'
        }
    
    Returns:
        JSON con la figura de Plotly
    """
    try:
        data = request.get_json()
        traces_paths = data.get('traces', {})
        metric = data.get('metric', 'nodes_explored')
        
        if not traces_paths:
            return jsonify({'error': 'traces is required'}), 400
        
        # Cargar todos los traces
        traces_dfs = {}
        for name, path in traces_paths.items():
            traces_dfs[name] = load_trace(path)
        
        # Generar comparación
        fig = compare_traces(traces_dfs, metric=metric)
        
        return jsonify({
            'success': True,
            'figure': json.loads(fig.to_json())
        })
    
    except FileNotFoundError as e:
        return jsonify({'error': f'Trace file not found: {str(e)}'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/report', methods=['POST'])
def generate_report_endpoint():
    """
    Genera un reporte HTML completo.
    
    Request Body:
        {
            "trace_path": str,
            "title": str,           # Opcional
            "advanced": bool        # Opcional, usar reporte avanzado
        }
    
    Returns:
        Archivo HTML del reporte
    """
    try:
        data = request.get_json()
        trace_path = data.get('trace_path')
        title = data.get('title', 'Reporte de Búsqueda')
        advanced = data.get('advanced', False)
        
        if not trace_path:
            return jsonify({'error': 'trace_path is required'}), 400
        
        # Cargar trace
        df = load_trace(trace_path)
        
        # Generar reporte en archivo temporal
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name
        
        if advanced:
            generate_advanced_report(df, output_path, title=title)
        else:
            generate_report(df, output_path, title=title)
        
        # Enviar archivo
        return send_file(
            output_path,
            mimetype='text/html',
            as_attachment=True,
            download_name='report.html'
        )
    
    except FileNotFoundError:
        return jsonify({'error': 'Trace file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/statistics', methods=['POST'])
def get_statistics():
    app.logger.info("Received request for /api/v1/statistics")
    data = request.get_json()
    trace_path = data.get("trace_path")

    if not trace_path:
        app.logger.error("trace_path is required")
        return jsonify({"error": "trace_path es requerido"}), 400

    try:
        app.logger.info(f"Loading trace from: {trace_path}")
        # Asegurarse de que el tracer se inicialice correctamente
        # No es necesario inicializar SearchSpaceTracer aquí, load_trace ya lo hace.
        df = load_trace(trace_path)
        app.logger.info(f"Trace loaded. DataFrame shape: {df.shape}")
        
        # Calcular estadísticas
        # Calcular estadísticas
        total_events = len(df)
        nodes_explored = len(df[df["event_type"] == "variable_assigned"])
        backtracks = len(df[df["event_type"] == "backtrack"])
        solutions = len(df[df["event_type"] == "solution_found"])
        ac3_calls = len(df[df["event_type"] == "ac3_call"])
        domain_prunes = len(df[df["event_type"] == "domain_pruned"])
        
        backtrack_rate = backtracks / nodes_explored if nodes_explored > 0 else 0
        
        duration = 0.0
        if len(df) >= 2:
            start_time = df["timestamp"].min()
            end_time = df["timestamp"].max()
            duration = (end_time - start_time).total_seconds() # Convertir Timedelta a segundos y luego a float
        elif len(df) == 1:

            # Si solo hay un evento, la duración es 0
            duration = 0.0
        
        max_depth = df["depth"].max() if len(df) > 0 else 0
        
        app.logger.info(f"Calculated statistics: nodes={nodes_explored}, backtracks={backtracks}, duration={duration}")

        return jsonify({
            "success": True,
            "statistics": {
                "total_events": int(total_events),
                "nodes_explored": int(nodes_explored),
                "backtracks": int(backtracks),
                "solutions": int(solutions),
                "ac3_calls": int(ac3_calls),
                "domain_prunes": int(domain_prunes),
                "backtrack_rate": float(backtrack_rate),
                "duration": float(duration),
                "max_depth": int(max_depth),
                "events_per_second": float(total_events / duration) if duration > 0 else 0
            }
        }), 200
    except FileNotFoundError:
        app.logger.error(f"Trace file not found: {trace_path}")
        return jsonify({"error": "Trace file not found"}), 404
    except Exception as e:
        app.logger.exception(f"Error processing statistics request: {e}")
        return jsonify({"error": str(e)}), 500


def run_api(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """
    Ejecuta el servidor API.
    
    Args:
        host: Host donde escuchar
        port: Puerto donde escuchar
        debug: Modo debug
    """
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_api(debug=True)


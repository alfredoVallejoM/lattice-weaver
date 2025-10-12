"""
Módulo de análisis de resultados de experimentos.

Proporciona funciones para analizar y visualizar resultados de experimentos.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

from typing import Dict, Any, List
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def analyze_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analiza resultados de experimentos.
    
    Args:
        results_df: DataFrame con los resultados
        
    Returns:
        Diccionario con análisis agregados
    """
    analysis = {}
    
    # Filtrar solo exitosos
    successful = results_df[results_df['success'] == True]
    
    if len(successful) == 0:
        return {'error': 'No hay ejecuciones exitosas para analizar'}
    
    # Estadísticas globales
    analysis['global'] = {
        'total_runs': len(results_df),
        'successful_runs': len(successful),
        'success_rate': len(successful) / len(results_df),
        'avg_time': successful['time_elapsed'].mean(),
        'std_time': successful['time_elapsed'].std(),
        'avg_nodes': successful['nodes_explored'].mean(),
        'std_nodes': successful['nodes_explored'].std()
    }
    
    # Estadísticas por configuración
    analysis['by_config'] = {}
    
    for config_name in results_df['config_name'].unique():
        config_df = results_df[results_df['config_name'] == config_name]
        config_successful = config_df[config_df['success'] == True]
        
        if len(config_successful) > 0:
            analysis['by_config'][config_name] = {
                'runs': len(config_df),
                'successful': len(config_successful),
                'success_rate': len(config_successful) / len(config_df),
                'avg_time': config_successful['time_elapsed'].mean(),
                'std_time': config_successful['time_elapsed'].std(),
                'min_time': config_successful['time_elapsed'].min(),
                'max_time': config_successful['time_elapsed'].max(),
                'avg_nodes': config_successful['nodes_explored'].mean(),
                'avg_backtracks': config_successful['backtracks'].mean()
            }
    
    return analysis


def generate_comparison_report(
    results_df: pd.DataFrame,
    output_path: str,
    title: str = "Reporte de Comparación de Experimentos"
) -> None:
    """
    Genera un reporte HTML comparando resultados de experimentos.
    
    Args:
        results_df: DataFrame con los resultados
        output_path: Ruta del archivo HTML de salida
        title: Título del reporte
    """
    # Filtrar solo exitosos
    successful = results_df[results_df['success'] == True]
    
    if len(successful) == 0:
        print("No hay resultados exitosos para generar el reporte")
        return
    
    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Tiempo de Ejecución por Configuración',
            'Nodos Explorados por Configuración',
            'Backtracks por Configuración',
            'Distribución de Tiempos'
        )
    )
    
    # Agrupar por configuración
    configs = successful['config_name'].unique()
    
    # Gráfico 1: Tiempo de ejecución
    for config in configs:
        config_data = successful[successful['config_name'] == config]
        fig.add_trace(
            go.Box(y=config_data['time_elapsed'], name=config),
            row=1, col=1
        )
    
    # Gráfico 2: Nodos explorados
    for config in configs:
        config_data = successful[successful['config_name'] == config]
        fig.add_trace(
            go.Box(y=config_data['nodes_explored'], name=config, showlegend=False),
            row=1, col=2
        )
    
    # Gráfico 3: Backtracks
    for config in configs:
        config_data = successful[successful['config_name'] == config]
        fig.add_trace(
            go.Box(y=config_data['backtracks'], name=config, showlegend=False),
            row=2, col=1
        )
    
    # Gráfico 4: Histograma de tiempos
    fig.add_trace(
        go.Histogram(x=successful['time_elapsed'], name='Distribución', showlegend=False),
        row=2, col=2
    )
    
    # Actualizar layout
    fig.update_xaxes(title_text="Configuración", row=1, col=1)
    fig.update_yaxes(title_text="Tiempo (s)", row=1, col=1)
    
    fig.update_xaxes(title_text="Configuración", row=1, col=2)
    fig.update_yaxes(title_text="Nodos", row=1, col=2)
    
    fig.update_xaxes(title_text="Configuración", row=2, col=1)
    fig.update_yaxes(title_text="Backtracks", row=2, col=1)
    
    fig.update_xaxes(title_text="Tiempo (s)", row=2, col=2)
    fig.update_yaxes(title_text="Frecuencia", row=2, col=2)
    
    fig.update_layout(
        title_text=title,
        height=800,
        showlegend=True
    )
    
    # Guardar
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_file))
    
    print(f"Reporte de comparación generado: {output_path}")





def compute_statistics_with_confidence(
    results_df: pd.DataFrame,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Calcula estadísticas con intervalos de confianza.
    
    Args:
        results_df: DataFrame con los resultados
        confidence_level: Nivel de confianza (default: 0.95 para 95%)
        
    Returns:
        Diccionario con estadísticas e intervalos de confianza
    """
    import scipy.stats as stats
    
    successful = results_df[results_df['success'] == True]
    
    if len(successful) == 0:
        return {'error': 'No hay ejecuciones exitosas'}
    
    def confidence_interval(data, confidence=0.95):
        """Calcula intervalo de confianza."""
        n = len(data)
        mean = data.mean()
        se = stats.sem(data)
        margin = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return (mean - margin, mean + margin)
    
    statistics = {}
    
    # Tiempo de ejecución
    time_data = successful['time_elapsed']
    statistics['time'] = {
        'mean': time_data.mean(),
        'median': time_data.median(),
        'std': time_data.std(),
        'min': time_data.min(),
        'max': time_data.max(),
        'percentile_25': time_data.quantile(0.25),
        'percentile_75': time_data.quantile(0.75),
        'confidence_interval': confidence_interval(time_data, confidence_level)
    }
    
    # Nodos explorados
    nodes_data = successful['nodes_explored']
    statistics['nodes'] = {
        'mean': nodes_data.mean(),
        'median': nodes_data.median(),
        'std': nodes_data.std(),
        'min': nodes_data.min(),
        'max': nodes_data.max(),
        'percentile_25': nodes_data.quantile(0.25),
        'percentile_75': nodes_data.quantile(0.75),
        'confidence_interval': confidence_interval(nodes_data, confidence_level)
    }
    
    # Backtracks
    backtracks_data = successful['backtracks']
    statistics['backtracks'] = {
        'mean': backtracks_data.mean(),
        'median': backtracks_data.median(),
        'std': backtracks_data.std(),
        'min': backtracks_data.min(),
        'max': backtracks_data.max(),
        'percentile_25': backtracks_data.quantile(0.25),
        'percentile_75': backtracks_data.quantile(0.75),
        'confidence_interval': confidence_interval(backtracks_data, confidence_level)
    }
    
    return statistics


def detect_outliers(results_df: pd.DataFrame, column: str = 'time_elapsed') -> pd.DataFrame:
    """
    Detecta outliers usando el método IQR.
    
    Args:
        results_df: DataFrame con los resultados
        column: Columna a analizar
        
    Returns:
        DataFrame con los outliers detectados
    """
    successful = results_df[results_df['success'] == True]
    
    if len(successful) == 0:
        return pd.DataFrame()
    
    Q1 = successful[column].quantile(0.25)
    Q3 = successful[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = successful[
        (successful[column] < lower_bound) | (successful[column] > upper_bound)
    ]
    
    return outliers


def generate_detailed_report(
    results_df: pd.DataFrame,
    output_path: str,
    title: str = "Reporte Detallado de Experimentos"
) -> None:
    """
    Genera un reporte HTML detallado con análisis avanzado.
    
    Args:
        results_df: DataFrame con los resultados
        output_path: Ruta del archivo HTML de salida
        title: Título del reporte
    """
    successful = results_df[results_df['success'] == True]
    
    if len(successful) == 0:
        print("No hay resultados exitosos para generar el reporte")
        return
    
    # Calcular estadísticas
    stats = compute_statistics_with_confidence(results_df)
    
    # Detectar outliers
    time_outliers = detect_outliers(results_df, 'time_elapsed')
    nodes_outliers = detect_outliers(results_df, 'nodes_explored')
    
    # Crear visualizaciones
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Distribución de Tiempos',
            'Distribución de Nodos',
            'Tiempo vs Nodos',
            'Backtracks vs Nodos',
            'Tasa de Éxito por Configuración',
            'Tiempo Promedio por Configuración'
        ),
        specs=[
            [{"type": "histogram"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}]
        ]
    )
    
    # Histograma de tiempos
    fig.add_trace(
        go.Histogram(x=successful['time_elapsed'], name='Tiempo', nbinsx=20),
        row=1, col=1
    )
    
    # Histograma de nodos
    fig.add_trace(
        go.Histogram(x=successful['nodes_explored'], name='Nodos', nbinsx=20),
        row=1, col=2
    )
    
    # Scatter: Tiempo vs Nodos
    fig.add_trace(
        go.Scatter(
            x=successful['nodes_explored'],
            y=successful['time_elapsed'],
            mode='markers',
            name='Ejecuciones',
            marker=dict(size=8, opacity=0.6)
        ),
        row=2, col=1
    )
    
    # Scatter: Backtracks vs Nodos
    fig.add_trace(
        go.Scatter(
            x=successful['nodes_explored'],
            y=successful['backtracks'],
            mode='markers',
            name='Ejecuciones',
            marker=dict(size=8, opacity=0.6),
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Tasa de éxito por configuración
    config_success = results_df.groupby('config_name').apply(
        lambda x: (x['success'] == True).sum() / len(x)
    )
    
    fig.add_trace(
        go.Bar(x=config_success.index, y=config_success.values, name='Tasa de Éxito'),
        row=3, col=1
    )
    
    # Tiempo promedio por configuración
    config_time = successful.groupby('config_name')['time_elapsed'].mean()
    
    fig.add_trace(
        go.Bar(x=config_time.index, y=config_time.values, name='Tiempo Promedio'),
        row=3, col=2
    )
    
    # Actualizar layout
    fig.update_layout(
        title_text=title,
        height=1200,
        showlegend=True
    )
    
    # Generar HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 20px auto;
            background-color: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: #f9f9f9;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            border-radius: 5px;
        }}
        .stat-box h3 {{
            margin-top: 0;
            color: #4CAF50;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
        }}
        .stat-label {{
            font-weight: bold;
        }}
        .outlier-box {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <h2>Resumen General</h2>
        <div class="stats-grid">
            <div class="stat-box">
                <h3>Tiempo de Ejecución</h3>
                <div class="stat-row">
                    <span class="stat-label">Media:</span>
                    <span>{stats['time']['mean']:.4f}s</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Mediana:</span>
                    <span>{stats['time']['median']:.4f}s</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Desv. Estándar:</span>
                    <span>{stats['time']['std']:.4f}s</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">IC 95%:</span>
                    <span>[{stats['time']['confidence_interval'][0]:.4f}, {stats['time']['confidence_interval'][1]:.4f}]</span>
                </div>
            </div>
            
            <div class="stat-box">
                <h3>Nodos Explorados</h3>
                <div class="stat-row">
                    <span class="stat-label">Media:</span>
                    <span>{stats['nodes']['mean']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Mediana:</span>
                    <span>{stats['nodes']['median']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Desv. Estándar:</span>
                    <span>{stats['nodes']['std']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">IC 95%:</span>
                    <span>[{stats['nodes']['confidence_interval'][0]:.1f}, {stats['nodes']['confidence_interval'][1]:.1f}]</span>
                </div>
            </div>
            
            <div class="stat-box">
                <h3>Backtracks</h3>
                <div class="stat-row">
                    <span class="stat-label">Media:</span>
                    <span>{stats['backtracks']['mean']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Mediana:</span>
                    <span>{stats['backtracks']['median']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Desv. Estándar:</span>
                    <span>{stats['backtracks']['std']:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">IC 95%:</span>
                    <span>[{stats['backtracks']['confidence_interval'][0]:.1f}, {stats['backtracks']['confidence_interval'][1]:.1f}]</span>
                </div>
            </div>
        </div>
        
        <h2>Outliers Detectados</h2>
        <div class="outlier-box">
            <p><strong>Outliers en Tiempo:</strong> {len(time_outliers)} ejecuciones</p>
            <p><strong>Outliers en Nodos:</strong> {len(nodes_outliers)} ejecuciones</p>
        </div>
        
        <h2>Visualizaciones</h2>
        <div id="visualizations"></div>
        
        <h2>Resultados Detallados</h2>
        <table>
            <thead>
                <tr>
                    <th>Configuración</th>
                    <th>Run ID</th>
                    <th>Tiempo (s)</th>
                    <th>Nodos</th>
                    <th>Backtracks</th>
                    <th>Soluciones</th>
                </tr>
            </thead>
            <tbody>
                {"".join([f'''
                <tr>
                    <td>{row['config_name']}</td>
                    <td>{row['run_id']}</td>
                    <td>{row['time_elapsed']:.4f}</td>
                    <td>{row['nodes_explored']}</td>
                    <td>{row['backtracks']}</td>
                    <td>{row['solutions_found']}</td>
                </tr>
                ''' for _, row in successful.iterrows()])}
            </tbody>
        </table>
    </div>
    
    <script>
        var data = {fig.to_json()};
        Plotly.newPlot('visualizations', data.data, data.layout);
    </script>
</body>
</html>
"""
    
    # Guardar
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_content, encoding='utf-8')
    
    print(f"Reporte detallado generado: {output_path}")


def export_results_to_csv(results_df: pd.DataFrame, output_path: str):
    """
    Exporta resultados a CSV.
    
    Args:
        results_df: DataFrame con los resultados
        output_path: Ruta del archivo CSV de salida
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_file, index=False)
    print(f"Resultados exportados a CSV: {output_path}")


def export_summary_to_markdown(
    results_df: pd.DataFrame,
    output_path: str,
    title: str = "Resumen de Experimentos"
):
    """
    Exporta un resumen en formato Markdown.
    
    Args:
        results_df: DataFrame con los resultados
        output_path: Ruta del archivo Markdown de salida
        title: Título del resumen
    """
    successful = results_df[results_df['success'] == True]
    
    if len(successful) == 0:
        print("No hay resultados exitosos para generar el resumen")
        return
    
    stats = compute_statistics_with_confidence(results_df)
    
    markdown = f"""# {title}

## Resumen General

- **Total de ejecuciones:** {len(results_df)}
- **Ejecuciones exitosas:** {len(successful)}
- **Tasa de éxito:** {len(successful) / len(results_df):.1%}

## Estadísticas de Tiempo

| Métrica | Valor |
|---------|-------|
| Media | {stats['time']['mean']:.4f}s |
| Mediana | {stats['time']['median']:.4f}s |
| Desviación Estándar | {stats['time']['std']:.4f}s |
| Mínimo | {stats['time']['min']:.4f}s |
| Máximo | {stats['time']['max']:.4f}s |
| Percentil 25 | {stats['time']['percentile_25']:.4f}s |
| Percentil 75 | {stats['time']['percentile_75']:.4f}s |
| IC 95% | [{stats['time']['confidence_interval'][0]:.4f}, {stats['time']['confidence_interval'][1]:.4f}] |

## Estadísticas de Nodos

| Métrica | Valor |
|---------|-------|
| Media | {stats['nodes']['mean']:.1f} |
| Mediana | {stats['nodes']['median']:.1f} |
| Desviación Estándar | {stats['nodes']['std']:.1f} |
| Mínimo | {stats['nodes']['min']} |
| Máximo | {stats['nodes']['max']} |
| Percentil 25 | {stats['nodes']['percentile_25']:.1f} |
| Percentil 75 | {stats['nodes']['percentile_75']:.1f} |
| IC 95% | [{stats['nodes']['confidence_interval'][0]:.1f}, {stats['nodes']['confidence_interval'][1]:.1f}] |

## Resultados por Configuración

"""
    
    for config_name in results_df['config_name'].unique():
        config_df = successful[successful['config_name'] == config_name]
        
        if len(config_df) > 0:
            markdown += f"""### {config_name}

- **Ejecuciones:** {len(config_df)}
- **Tiempo promedio:** {config_df['time_elapsed'].mean():.4f}s ± {config_df['time_elapsed'].std():.4f}s
- **Nodos promedio:** {config_df['nodes_explored'].mean():.1f} ± {config_df['nodes_explored'].std():.1f}
- **Backtracks promedio:** {config_df['backtracks'].mean():.1f} ± {config_df['backtracks'].std():.1f}

"""
    
    # Guardar
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(markdown, encoding='utf-8')
    
    print(f"Resumen exportado a Markdown: {output_path}")


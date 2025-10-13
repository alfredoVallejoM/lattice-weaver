"""
SearchSpaceVisualizer - Visualización del Espacio de Búsqueda.

Este módulo proporciona funciones para visualizar traces de búsqueda
generados por el SearchSpaceTracer, creando gráficos interactivos y
reportes HTML completos.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

from typing import Optional
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def load_trace(path: str) -> pd.DataFrame:
    """
    Carga un archivo de trace automáticamente detectando el formato.
    
    Args:
        path: Ruta del archivo de trace (.csv o .jsonl)
        
    Returns:
        DataFrame de pandas con los eventos
        
    Raises:
        ValueError: Si el formato no es soportado
        FileNotFoundError: Si el archivo no existe
        
    Examples:
        >>> df = load_trace("trace.csv")
        >>> print(df.shape)
        (100, 8)
    """
    from lattice_weaver.core.csp_engine.tracing import load_trace as _load_trace
    return _load_trace(path)


def plot_search_tree(trace_df: pd.DataFrame, max_nodes: int = 1000) -> go.Figure:
    """
    Genera una visualización del árbol de búsqueda usando un gráfico icicle.
    
    El gráfico muestra la estructura jerárquica del árbol de búsqueda,
    donde cada nivel representa una profundidad y cada rectángulo
    representa una asignación de variable.
    
    Args:
        trace_df: DataFrame con los eventos de trace
        max_nodes: Número máximo de nodos a visualizar (para rendimiento)
        
    Returns:
        Figura de Plotly con el gráfico icicle
        
    Examples:
        >>> df = load_trace("trace.csv")
        >>> fig = plot_search_tree(df)
        >>> fig.show()
    """
    # Filtrar solo eventos de asignación y backtrack
    events = trace_df[
        trace_df['event_type'].isin(['variable_assigned', 'backtrack', 'solution_found'])
    ].copy()
    
    if len(events) == 0:
        # Crear figura vacía si no hay eventos
        fig = go.Figure()
        fig.add_annotation(
            text="No hay eventos de búsqueda para visualizar",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Limitar número de nodos para rendimiento
    if len(events) > max_nodes:
        events = events.head(max_nodes)
    
    # Construir estructura del árbol
    labels = ["Root"]
    parents = [""]
    values = [1]
    colors = [0]
    hover_texts = ["Inicio de búsqueda"]
    
    # Rastrear el camino actual en el árbol
    current_path = []
    node_counter = 0
    
    for idx, row in events.iterrows():
        event_type = row['event_type']
        depth = row['depth']
        variable = row['variable']
        value = row['value']
        
        if event_type == 'variable_assigned':
            # Ajustar el camino actual a la profundidad
            current_path = current_path[:depth]
            
            # Crear nodo
            node_id = f"node_{node_counter}"
            node_counter += 1
            
            # Determinar padre
            if len(current_path) == 0:
                parent = "Root"
            else:
                parent = current_path[-1]
            
            # Añadir nodo
            label = f"{variable}={value}"
            labels.append(label)
            parents.append(parent)
            values.append(1)
            colors.append(depth)
            hover_texts.append(f"{variable} = {value}<br>Profundidad: {depth}")
            
            # Actualizar camino
            current_path.append(node_id)
            
        elif event_type == 'backtrack':
            # Retroceder en el camino
            if len(current_path) > 0:
                current_path.pop()
        
        elif event_type == 'solution_found':
            # Marcar solución
            if len(current_path) > 0:
                parent = current_path[-1]
            else:
                parent = "Root"
            
            labels.append("✓ Solución")
            parents.append(parent)
            values.append(1)
            colors.append(depth + 1)
            hover_texts.append("Solución encontrada")
    
    # Crear gráfico icicle
    fig = go.Figure(go.Icicle(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colorscale='Viridis',
            cmid=max(colors) / 2 if colors else 0,
            colorbar=dict(title="Profundidad")
        ),
        text=hover_texts,
        hovertemplate='<b>%{label}</b><br>%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Árbol de Búsqueda",
        height=600,
        margin=dict(t=50, l=0, r=0, b=0)
    )
    
    return fig


def plot_domain_evolution(trace_df: pd.DataFrame) -> go.Figure:
    """
    Muestra cómo el tamaño de los dominios cambia con el tiempo.
    
    Esta visualización ayuda a identificar variables que son difíciles
    de resolver (dominios que se reducen lentamente) y momentos de
    poda intensa.
    
    Args:
        trace_df: DataFrame con los eventos de trace
        
    Returns:
        Figura de Plotly con gráfico de líneas
        
    Examples:
        >>> df = load_trace("trace.csv")
        >>> fig = plot_domain_evolution(df)
        >>> fig.show()
    """
    # Filtrar eventos de poda de dominio
    prune_events = trace_df[trace_df['event_type'] == 'domain_pruned'].copy()
    
    if len(prune_events) == 0:
        # Crear figura vacía si no hay eventos de poda
        fig = go.Figure()
        fig.add_annotation(
            text="No hay eventos de poda de dominio para visualizar",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Calcular tiempo relativo
    if len(trace_df) > 0:
        start_time = trace_df['timestamp'].min()
        prune_events['relative_time'] = prune_events['timestamp'] - start_time
    else:
        prune_events['relative_time'] = 0
    
    # Contar podas por variable a lo largo del tiempo
    prune_counts = prune_events.groupby(['variable', 'relative_time']).size().reset_index(name='prune_count')
    
    # Crear gráfico de líneas
    fig = px.line(
        prune_counts,
        x='relative_time',
        y='prune_count',
        color='variable',
        title='Evolución de Podas de Dominio',
        labels={
            'relative_time': 'Tiempo (s)',
            'prune_count': 'Número de Podas',
            'variable': 'Variable'
        }
    )
    
    fig.update_layout(
        height=500,
        hovermode='x unified'
    )
    
    return fig


def plot_backtrack_heatmap(trace_df: pd.DataFrame) -> go.Figure:
    """
    Genera un heatmap mostrando en qué variables y profundidades ocurren más backtracks.
    
    Esta visualización ayuda a identificar cuellos de botella en la búsqueda
    y variables problemáticas.
    
    Args:
        trace_df: DataFrame con los eventos de trace
        
    Returns:
        Figura de Plotly con heatmap
        
    Examples:
        >>> df = load_trace("trace.csv")
        >>> fig = plot_backtrack_heatmap(df)
        >>> fig.show()
    """
    # Filtrar eventos de backtrack
    backtrack_events = trace_df[trace_df['event_type'] == 'backtrack'].copy()
    
    if len(backtrack_events) == 0:
        # Crear figura vacía si no hay backtracks
        fig = go.Figure()
        fig.add_annotation(
            text="No hay eventos de backtrack para visualizar",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Contar backtracks por variable y profundidad
    heatmap_data = backtrack_events.groupby(['variable', 'depth']).size().reset_index(name='count')
    
    # Crear matriz para el heatmap
    pivot_table = heatmap_data.pivot(index='variable', columns='depth', values='count')
    pivot_table = pivot_table.fillna(0)
    
    # Crear heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='Reds',
        hovertemplate='Variable: %{y}<br>Profundidad: %{x}<br>Backtracks: %{z}<extra></extra>',
        colorbar=dict(title="Backtracks")
    ))
    
    fig.update_layout(
        title='Heatmap de Backtracks',
        xaxis_title='Profundidad',
        yaxis_title='Variable',
        height=max(400, len(pivot_table.index) * 30)
    )
    
    return fig


def generate_report(
    trace_df: pd.DataFrame,
    output_path: str,
    title: str = "Reporte de Búsqueda"
) -> None:
    """
    Genera un reporte HTML completo con todas las visualizaciones y estadísticas.
    
    El reporte incluye:
    - Resumen de estadísticas
    - Árbol de búsqueda
    - Evolución de dominios
    - Heatmap de backtracks
    
    Args:
        trace_df: DataFrame con los eventos de trace
        output_path: Ruta del archivo HTML de salida
        title: Título del reporte
        
    Examples:
        >>> df = load_trace("trace.csv")
        >>> generate_report(df, "report.html", title="N-Reinas 8x8")
    """
    # Calcular estadísticas
    total_events = len(trace_df)
    nodes_explored = len(trace_df[trace_df['event_type'] == 'variable_assigned'])
    backtracks = len(trace_df[trace_df['event_type'] == 'backtrack'])
    solutions = len(trace_df[trace_df['event_type'] == 'solution_found'])
    
    backtrack_rate = backtracks / nodes_explored if nodes_explored > 0 else 0
    
    if len(trace_df) >= 2:
        start_time = trace_df['timestamp'].min()
        end_time = trace_df['timestamp'].max()
        duration = end_time - start_time
    else:
        duration = 0
    
    # Generar visualizaciones
    fig_tree = plot_search_tree(trace_df)
    fig_domain = plot_domain_evolution(trace_df)
    fig_heatmap = plot_backtrack_heatmap(trace_df)
    
    # Crear HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
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
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .visualization {{
            margin: 30px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            background-color: #fafafa;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #777;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <h2>Resumen de Estadísticas</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total de Eventos</div>
                <div class="stat-value">{total_events}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Nodos Explorados</div>
                <div class="stat-value">{nodes_explored}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Backtracks</div>
                <div class="stat-value">{backtracks}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Soluciones</div>
                <div class="stat-value">{solutions}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Tasa de Backtrack</div>
                <div class="stat-value">{backtrack_rate:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Duración</div>
                <div class="stat-value">{duration:.3f}s</div>
            </div>
        </div>
        
        <h2>Árbol de Búsqueda</h2>
        <div class="visualization" id="tree"></div>
        
        <h2>Evolución de Podas de Dominio</h2>
        <div class="visualization" id="domain"></div>
        
        <h2>Heatmap de Backtracks</h2>
        <div class="visualization" id="heatmap"></div>
        
        <div class="footer">
            Generado por LatticeWeaver SearchSpaceVisualizer<br>
            Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
    
    <script>
        // Árbol de búsqueda
        var treeData = {fig_tree.to_json()};
        Plotly.newPlot('tree', treeData.data, treeData.layout);
        
        // Evolución de dominios
        var domainData = {fig_domain.to_json()};
        Plotly.newPlot('domain', domainData.data, domainData.layout);
        
        // Heatmap
        var heatmapData = {fig_heatmap.to_json()};
        Plotly.newPlot('heatmap', heatmapData.data, heatmapData.layout);
    </script>
</body>
</html>
"""
    
    # Guardar HTML
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_content, encoding='utf-8')
    
    print(f"Reporte generado: {output_path}")





def plot_timeline(trace_df: pd.DataFrame) -> go.Figure:
    """
    Genera una visualización de línea de tiempo de los eventos de búsqueda.
    
    Muestra cuándo ocurren diferentes tipos de eventos a lo largo del tiempo,
    útil para identificar patrones temporales y fases de la búsqueda.
    
    Args:
        trace_df: DataFrame con los eventos de trace
        
    Returns:
        Figura de Plotly con gráfico de línea de tiempo
        
    Examples:
        >>> df = load_trace("trace.csv")
        >>> fig = plot_timeline(df)
        >>> fig.show()
    """
    if len(trace_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No hay eventos para visualizar",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Calcular tiempo relativo
    start_time = trace_df['timestamp'].min()
    trace_df_copy = trace_df.copy()
    trace_df_copy['relative_time'] = trace_df_copy['timestamp'] - start_time
    
    # Contar eventos por tipo en ventanas de tiempo
    event_counts = trace_df_copy.groupby(['event_type']).size().reset_index(name='count')
    
    # Crear gráfico de dispersión con líneas
    fig = go.Figure()
    
    for event_type in trace_df_copy['event_type'].unique():
        events = trace_df_copy[trace_df_copy['event_type'] == event_type]
        
        fig.add_trace(go.Scatter(
            x=events['relative_time'],
            y=[event_type] * len(events),
            mode='markers',
            name=event_type,
            marker=dict(size=8),
            hovertemplate='%{x:.4f}s<extra></extra>'
        ))
    
    fig.update_layout(
        title='Línea de Tiempo de Eventos',
        xaxis_title='Tiempo (s)',
        yaxis_title='Tipo de Evento',
        height=500,
        hovermode='closest'
    )
    
    return fig


def plot_variable_statistics(trace_df: pd.DataFrame) -> go.Figure:
    """
    Genera estadísticas por variable (asignaciones, backtracks, podas).
    
    Muestra un gráfico de barras con métricas agregadas por variable,
    útil para identificar variables problemáticas.
    
    Args:
        trace_df: DataFrame con los eventos de trace
        
    Returns:
        Figura de Plotly con gráfico de barras
        
    Examples:
        >>> df = load_trace("trace.csv")
        >>> fig = plot_variable_statistics(df)
        >>> fig.show()
    """
    # Contar eventos por variable
    var_assignments = trace_df[trace_df['event_type'] == 'variable_assigned'].groupby('variable').size()
    var_backtracks = trace_df[trace_df['event_type'] == 'backtrack'].groupby('variable').size()
    var_prunes = trace_df[trace_df['event_type'] == 'domain_pruned'].groupby('variable').size()
    
    # Combinar en un DataFrame
    all_vars = set(var_assignments.index) | set(var_backtracks.index) | set(var_prunes.index)
    
    if len(all_vars) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No hay variables para visualizar",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    stats_data = []
    for var in sorted(all_vars):
        stats_data.append({
            'variable': var,
            'assignments': var_assignments.get(var, 0),
            'backtracks': var_backtracks.get(var, 0),
            'prunes': var_prunes.get(var, 0)
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Crear gráfico de barras agrupadas
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Asignaciones',
        x=stats_df['variable'],
        y=stats_df['assignments'],
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Backtracks',
        x=stats_df['variable'],
        y=stats_df['backtracks'],
        marker_color='salmon'
    ))
    
    fig.add_trace(go.Bar(
        name='Podas',
        x=stats_df['variable'],
        y=stats_df['prunes'],
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title='Estadísticas por Variable',
        xaxis_title='Variable',
        yaxis_title='Cantidad',
        barmode='group',
        height=500
    )
    
    return fig


def compare_traces(trace_dfs: dict, metric: str = 'nodes_explored') -> go.Figure:
    """
    Compara múltiples traces en una visualización.
    
    Útil para comparar diferentes estrategias de búsqueda, heurísticas
    o configuraciones de parámetros.
    
    Args:
        trace_dfs: Diccionario {nombre: DataFrame} con los traces a comparar
        metric: Métrica a comparar ('nodes_explored', 'backtracks', 'time')
        
    Returns:
        Figura de Plotly con gráfico comparativo
        
    Examples:
        >>> df1 = load_trace("trace1.csv")
        >>> df2 = load_trace("trace2.csv")
        >>> fig = compare_traces({'Estrategia A': df1, 'Estrategia B': df2})
        >>> fig.show()
    """
    comparison_data = []
    
    for name, df in trace_dfs.items():
        nodes = len(df[df['event_type'] == 'variable_assigned'])
        backtracks = len(df[df['event_type'] == 'backtrack'])
        solutions = len(df[df['event_type'] == 'solution_found'])
        
        if len(df) >= 2:
            time_elapsed = df['timestamp'].max() - df['timestamp'].min()
        else:
            time_elapsed = 0
        
        comparison_data.append({
            'name': name,
            'nodes_explored': nodes,
            'backtracks': backtracks,
            'solutions': solutions,
            'time': time_elapsed,
            'backtrack_rate': backtracks / nodes if nodes > 0 else 0
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Crear gráfico de barras
    if metric == 'time':
        y_data = comp_df['time']
        y_label = 'Tiempo (s)'
    elif metric == 'backtracks':
        y_data = comp_df['backtracks']
        y_label = 'Backtracks'
    elif metric == 'backtrack_rate':
        y_data = comp_df['backtrack_rate']
        y_label = 'Tasa de Backtrack'
    else:  # nodes_explored
        y_data = comp_df['nodes_explored']
        y_label = 'Nodos Explorados'
    
    fig = go.Figure(data=[
        go.Bar(
            x=comp_df['name'],
            y=y_data,
            marker_color='steelblue',
            text=y_data,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f'Comparación de Traces - {y_label}',
        xaxis_title='Configuración',
        yaxis_title=y_label,
        height=500
    )
    
    return fig


def export_visualizations(
    trace_df: pd.DataFrame,
    output_dir: str,
    formats: list = ['html', 'png']
) -> dict:
    """
    Exporta todas las visualizaciones en los formatos especificados.
    
    Args:
        trace_df: DataFrame con los eventos de trace
        output_dir: Directorio de salida
        formats: Lista de formatos ('html', 'png', 'pdf', 'svg')
        
    Returns:
        Diccionario con las rutas de los archivos generados
        
    Examples:
        >>> df = load_trace("trace.csv")
        >>> files = export_visualizations(df, "output/", formats=['html', 'png'])
        >>> print(files)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_files = {}
    
    # Generar todas las visualizaciones
    visualizations = {
        'search_tree': plot_search_tree(trace_df),
        'domain_evolution': plot_domain_evolution(trace_df),
        'backtrack_heatmap': plot_backtrack_heatmap(trace_df),
        'timeline': plot_timeline(trace_df),
        'variable_stats': plot_variable_statistics(trace_df)
    }
    
    # Exportar en cada formato
    for viz_name, fig in visualizations.items():
        generated_files[viz_name] = {}
        
        for fmt in formats:
            file_path = output_path / f"{viz_name}.{fmt}"
            
            if fmt == 'html':
                fig.write_html(str(file_path))
            elif fmt == 'png':
                fig.write_image(str(file_path), width=1200, height=800)
            elif fmt == 'pdf':
                fig.write_image(str(file_path), width=1200, height=800)
            elif fmt == 'svg':
                fig.write_image(str(file_path), width=1200, height=800)
            
            generated_files[viz_name][fmt] = str(file_path)
    
    return generated_files


def generate_advanced_report(
    trace_df: pd.DataFrame,
    output_path: str,
    title: str = "Reporte Avanzado de Búsqueda",
    include_timeline: bool = True,
    include_variable_stats: bool = True
) -> None:
    """
    Genera un reporte HTML avanzado con todas las visualizaciones disponibles.
    
    Versión extendida de generate_report() con más visualizaciones y opciones.
    
    Args:
        trace_df: DataFrame con los eventos de trace
        output_path: Ruta del archivo HTML de salida
        title: Título del reporte
        include_timeline: Incluir visualización de línea de tiempo
        include_variable_stats: Incluir estadísticas por variable
        
    Examples:
        >>> df = load_trace("trace.csv")
        >>> generate_advanced_report(df, "advanced_report.html")
    """
    # Calcular estadísticas
    total_events = len(trace_df)
    nodes_explored = len(trace_df[trace_df['event_type'] == 'variable_assigned'])
    backtracks = len(trace_df[trace_df['event_type'] == 'backtrack'])
    solutions = len(trace_df[trace_df['event_type'] == 'solution_found'])
    ac3_calls = len(trace_df[trace_df['event_type'] == 'ac3_call'])
    domain_prunes = len(trace_df[trace_df['event_type'] == 'domain_pruned'])
    
    backtrack_rate = backtracks / nodes_explored if nodes_explored > 0 else 0
    
    if len(trace_df) >= 2:
        start_time = trace_df['timestamp'].min()
        end_time = trace_df['timestamp'].max()
        duration = end_time - start_time
    else:
        duration = 0
    
    # Generar visualizaciones
    fig_tree = plot_search_tree(trace_df)
    fig_domain = plot_domain_evolution(trace_df)
    fig_heatmap = plot_backtrack_heatmap(trace_df)
    
    timeline_html = ""
    if include_timeline:
        fig_timeline = plot_timeline(trace_df)
        timeline_html = f"""
        <h2>Línea de Tiempo de Eventos</h2>
        <div class="visualization" id="timeline"></div>
        <script>
            var timelineData = {fig_timeline.to_json()};
            Plotly.newPlot('timeline', timelineData.data, timelineData.layout);
        </script>
        """
    
    var_stats_html = ""
    if include_variable_stats:
        fig_var_stats = plot_variable_statistics(trace_df)
        var_stats_html = f"""
        <h2>Estadísticas por Variable</h2>
        <div class="visualization" id="varstats"></div>
        <script>
            var varStatsData = {fig_var_stats.to_json()};
            Plotly.newPlot('varstats', varStatsData.data, varStatsData.layout);
        </script>
        """
    
    # Crear HTML
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            max-width: 1400px;
            margin: 20px auto;
            background-color: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #333;
            border-bottom: 4px solid #667eea;
            padding-bottom: 15px;
            font-size: 2.5em;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #555;
            margin-top: 40px;
            font-size: 1.8em;
            border-left: 5px solid #764ba2;
            padding-left: 15px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 25px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 15px 0;
        }}
        .stat-label {{
            font-size: 1em;
            opacity: 0.95;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .visualization {{
            margin: 30px 0;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            padding: 25px;
            background-color: #fafafa;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 30px;
            border-top: 2px solid #e0e0e0;
            text-align: center;
            color: #777;
            font-size: 0.95em;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            background-color: #667eea;
            color: white;
            border-radius: 20px;
            font-size: 0.9em;
            margin: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div style="margin: 20px 0;">
            <span class="badge">Total: {total_events} eventos</span>
            <span class="badge">Duración: {duration:.3f}s</span>
            <span class="badge">Soluciones: {solutions}</span>
        </div>
        
        <h2>Resumen de Estadísticas</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Nodos Explorados</div>
                <div class="stat-value">{nodes_explored}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Backtracks</div>
                <div class="stat-value">{backtracks}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Tasa de Backtrack</div>
                <div class="stat-value">{backtrack_rate:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Llamadas AC-3</div>
                <div class="stat-value">{ac3_calls}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Podas de Dominio</div>
                <div class="stat-value">{domain_prunes}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Eventos/segundo</div>
                <div class="stat-value">{total_events/duration if duration > 0 else 0:.0f}</div>
            </div>
        </div>
        
        <h2>Árbol de Búsqueda</h2>
        <div class="visualization" id="tree"></div>
        
        {timeline_html}
        
        <h2>Evolución de Podas de Dominio</h2>
        <div class="visualization" id="domain"></div>
        
        <h2>Heatmap de Backtracks</h2>
        <div class="visualization" id="heatmap"></div>
        
        {var_stats_html}
        
        <div class="footer">
            <strong>Generado por LatticeWeaver SearchSpaceVisualizer</strong><br>
            Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <small>Track A - Core Engine & Visualization</small>
        </div>
    </div>
    
    <script>
        // Árbol de búsqueda
        var treeData = {fig_tree.to_json()};
        Plotly.newPlot('tree', treeData.data, treeData.layout);
        
        // Evolución de dominios
        var domainData = {fig_domain.to_json()};
        Plotly.newPlot('domain', domainData.data, domainData.layout);
        
        // Heatmap
        var heatmapData = {fig_heatmap.to_json()};
        Plotly.newPlot('heatmap', heatmapData.data, heatmapData.layout);
    </script>
</body>
</html>
"""
    
    # Guardar HTML
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_content, encoding='utf-8')
    
    print(f"Reporte avanzado generado: {output_path}")


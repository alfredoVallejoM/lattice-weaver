# lattice_weaver/visualization/core.py

"""
@i18n:key visualization_core_module
@i18n:category visualization
@i18n:desc_es Módulo core de visualización para LatticeWeaver. Proporciona funcionalidades base para renderizado de grafos, animaciones y exportación.
@i18n:desc_en Core visualization module for LatticeWeaver. Provides base functionality for graph rendering, animations, and export.
@i18n:desc_fr Module de visualisation de base pour LatticeWeaver. Fournit des fonctionnalités de base pour le rendu de graphes, les animations et l'exportation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum


class Theme(Enum):
    """
    @i18n:key theme_enum
    @i18n:desc_es Temas visuales disponibles para las visualizaciones.
    @i18n:desc_en Available visual themes for visualizations.
    @i18n:desc_fr Thèmes visuels disponibles pour les visualisations.
    """
    LIGHT = "light"
    DARK = "dark"
    PRESENTATION = "presentation"


class VisualizationEngine:
    """
    @i18n:key visualization_engine_class
    @i18n:desc_es Motor de visualización de alto rendimiento para grafos y estructuras matemáticas.
    @i18n:desc_en High-performance visualization engine for graphs and mathematical structures.
    @i18n:desc_fr Moteur de visualisation haute performance pour les graphes et les structures mathématiques.
    """
    
    # Paletas de colores por tema
    THEMES = {
        Theme.LIGHT: {
            "background": "#FFFFFF",
            "foreground": "#000000",
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "accent": "#F18F01",
            "success": "#06A77D",
            "warning": "#F77F00",
            "error": "#D62828",
            "node": "#E0E0E0",
            "edge": "#757575",
            "highlight": "#FFEB3B",
            "font_size": 10,
            "line_width": 1.5
        },
        Theme.DARK: {
            "background": "#1E1E1E",
            "foreground": "#FFFFFF",
            "primary": "#61AFEF",
            "secondary": "#C678DD",
            "accent": "#E5C07B",
            "success": "#98C379",
            "warning": "#D19A66",
            "error": "#E06C75",
            "node": "#3E4451",
            "edge": "#ABB2BF",
            "highlight": "#E5C07B",
            "font_size": 10,
            "line_width": 1.5
        },
        Theme.PRESENTATION: {
            "background": "#F5F5F5",
            "foreground": "#212121",
            "primary": "#1976D2",
            "secondary": "#7B1FA2",
            "accent": "#F57C00",
            "success": "#388E3C",
            "warning": "#FFA000",
            "error": "#D32F2F",
            "node": "#FFFFFF",
            "edge": "#424242",
            "highlight": "#FFEB3B",
            "font_size": 14,
            "line_width": 2.5
        }
    }
    
    def __init__(self, theme: Theme = Theme.LIGHT, figsize: Tuple[int, int] = (12, 8)):
        """
        @i18n:key visualization_engine_init
        @i18n:desc_es Inicializa el motor de visualización.
        @i18n:desc_en Initializes the visualization engine.
        @i18n:desc_fr Initialise le moteur de visualisation.
        
        Parameters
        ----------
        theme : Theme, default=Theme.LIGHT
            @i18n:param theme
            @i18n:type Theme
            @i18n:desc_es Tema visual a utilizar.
            @i18n:desc_en Visual theme to use.
            @i18n:desc_fr Thème visuel à utiliser.
        figsize : Tuple[int, int], default=(12, 8)
            @i18n:param figsize
            @i18n:type Tuple[int, int]
            @i18n:desc_es Tamaño de la figura en pulgadas (ancho, alto).
            @i18n:desc_en Figure size in inches (width, height).
            @i18n:desc_fr Taille de la figure en pouces (largeur, hauteur).
        """
        self.theme = theme
        self.colors = self.THEMES[theme]
        self.figsize = figsize
        self.fig = None
        self.ax = None
    
    def create_figure(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        @i18n:key create_figure
        @i18n:desc_es Crea una nueva figura de matplotlib con el tema configurado.
        @i18n:desc_en Creates a new matplotlib figure with the configured theme.
        @i18n:desc_fr Crée une nouvelle figure matplotlib avec le thème configuré.
        
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            @i18n:return tuple_fig_ax
            @i18n:desc_es Tupla con la figura y los ejes.
            @i18n:desc_en Tuple with the figure and axes.
            @i18n:desc_fr Tuple avec la figure et les axes.
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.fig.patch.set_facecolor(self.colors["background"])
        self.ax.set_facecolor(self.colors["background"])
        return self.fig, self.ax
    
    def render_graph(
        self,
        graph: nx.Graph,
        layout: str = "spring",
        node_labels: Optional[Dict] = None,
        edge_labels: Optional[Dict] = None,
        highlighted_nodes: Optional[List] = None,
        highlighted_edges: Optional[List] = None
    ) -> plt.Figure:
        """
        @i18n:key render_graph
        @i18n:desc_es Renderiza un grafo de NetworkX con el tema configurado.
        @i18n:desc_en Renders a NetworkX graph with the configured theme.
        @i18n:desc_fr Rend un graphe NetworkX avec le thème configuré.
        
        Parameters
        ----------
        graph : nx.Graph
            @i18n:param graph
            @i18n:type nx.Graph
            @i18n:desc_es Grafo a renderizar.
            @i18n:desc_en Graph to render.
            @i18n:desc_fr Graphe à rendre.
        layout : str, default="spring"
            @i18n:param layout
            @i18n:type str
            @i18n:values ["spring", "circular", "kamada_kawai", "spectral", "shell"]
            @i18n:desc_es Algoritmo de layout a utilizar.
            @i18n:desc_en Layout algorithm to use.
            @i18n:desc_fr Algorithme de disposition à utiliser.
        node_labels : Optional[Dict], default=None
            @i18n:param node_labels
            @i18n:type Optional[Dict]
            @i18n:desc_es Etiquetas personalizadas para nodos.
            @i18n:desc_en Custom labels for nodes.
            @i18n:desc_fr Étiquettes personnalisées pour les nœuds.
        edge_labels : Optional[Dict], default=None
            @i18n:param edge_labels
            @i18n:type Optional[Dict]
            @i18n:desc_es Etiquetas personalizadas para aristas.
            @i18n:desc_en Custom labels for edges.
            @i18n:desc_fr Étiquettes personnalisées pour les arêtes.
        highlighted_nodes : Optional[List], default=None
            @i18n:param highlighted_nodes
            @i18n:type Optional[List]
            @i18n:desc_es Lista de nodos a resaltar.
            @i18n:desc_en List of nodes to highlight.
            @i18n:desc_fr Liste de nœuds à mettre en évidence.
        highlighted_edges : Optional[List], default=None
            @i18n:param highlighted_edges
            @i18n:type Optional[List]
            @i18n:desc_es Lista de aristas a resaltar.
            @i18n:desc_en List of edges to highlight.
            @i18n:desc_fr Liste d'arêtes à mettre en évidence.
        
        Returns
        -------
        plt.Figure
            @i18n:return figure
            @i18n:desc_es Figura de matplotlib con el grafo renderizado.
            @i18n:desc_en Matplotlib figure with the rendered graph.
            @i18n:desc_fr Figure matplotlib avec le graphe rendu.
        """
        self.create_figure()
        
        # Calcular layout
        layout_funcs = {
            "spring": nx.spring_layout,
            "circular": nx.circular_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "spectral": nx.spectral_layout,
            "shell": nx.shell_layout
        }
        
        pos = layout_funcs.get(layout, nx.spring_layout)(graph)
        
        # Preparar colores de nodos
        node_colors = []
        for node in graph.nodes():
            if highlighted_nodes and node in highlighted_nodes:
                node_colors.append(self.colors["highlight"])
            else:
                node_colors.append(self.colors["node"])
        
        # Preparar colores de aristas
        edge_colors = []
        for edge in graph.edges():
            if highlighted_edges and edge in highlighted_edges:
                edge_colors.append(self.colors["highlight"])
            else:
                edge_colors.append(self.colors["edge"])
        
        # Dibujar nodos
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors,
            node_size=500,
            edgecolors=self.colors["foreground"],
            linewidths=self.colors["line_width"],
            ax=self.ax
        )
        
        # Dibujar aristas
        nx.draw_networkx_edges(
            graph, pos,
            edge_color=edge_colors,
            width=self.colors["line_width"],
            ax=self.ax
        )
        
        # Dibujar etiquetas de nodos
        if node_labels is None:
            node_labels = {node: str(node) for node in graph.nodes()}
        
        nx.draw_networkx_labels(
            graph, pos,
            labels=node_labels,
            font_size=self.colors["font_size"],
            font_color=self.colors["foreground"],
            ax=self.ax
        )
        
        # Dibujar etiquetas de aristas si se proporcionan
        if edge_labels:
            nx.draw_networkx_edge_labels(
                graph, pos,
                edge_labels=edge_labels,
                font_size=self.colors["font_size"] - 2,
                font_color=self.colors["foreground"],
                ax=self.ax
            )
        
        self.ax.axis('off')
        plt.tight_layout()
        
        return self.fig
    
    def export(self, filename: str, dpi: int = 300, format: str = "png"):
        """
        @i18n:key export
        @i18n:desc_es Exporta la figura actual a un archivo.
        @i18n:desc_en Exports the current figure to a file.
        @i18n:desc_fr Exporte la figure actuelle vers un fichier.
        
        Parameters
        ----------
        filename : str
            @i18n:param filename
            @i18n:type str
            @i18n:desc_es Nombre del archivo de salida.
            @i18n:desc_en Output filename.
            @i18n:desc_fr Nom du fichier de sortie.
        dpi : int, default=300
            @i18n:param dpi
            @i18n:type int
            @i18n:desc_es Resolución en puntos por pulgada.
            @i18n:desc_en Resolution in dots per inch.
            @i18n:desc_fr Résolution en points par pouce.
        format : str, default="png"
            @i18n:param format
            @i18n:type str
            @i18n:values ["png", "svg", "pdf", "jpg"]
            @i18n:desc_es Formato de salida.
            @i18n:desc_en Output format.
            @i18n:desc_fr Format de sortie.
        """
        if self.fig is None:
            raise ValueError("No figure to export. Call render_graph() first.")
        
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.fig.savefig(
            filename,
            dpi=dpi,
            format=format,
            facecolor=self.colors["background"],
            edgecolor='none',
            bbox_inches='tight'
        )
        
        print(f"✅ Figura exportada a: {filename}")
    
    def show(self):
        """
        @i18n:key show
        @i18n:desc_es Muestra la figura actual.
        @i18n:desc_en Shows the current figure.
        @i18n:desc_fr Affiche la figure actuelle.
        """
        if self.fig is None:
            raise ValueError("No figure to show. Call render_graph() first.")
        
        plt.show()
    
    def close(self):
        """
        @i18n:key close
        @i18n:desc_es Cierra la figura actual y libera recursos.
        @i18n:desc_en Closes the current figure and frees resources.
        @i18n:desc_fr Ferme la figure actuelle et libère les ressources.
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


def create_visualization_engine(theme: str = "light", **kwargs) -> VisualizationEngine:
    """
    @i18n:key create_visualization_engine
    @i18n:desc_es Función de conveniencia para crear un motor de visualización.
    @i18n:desc_en Convenience function to create a visualization engine.
    @i18n:desc_fr Fonction de commodité pour créer un moteur de visualisation.
    
    Parameters
    ----------
    theme : str, default="light"
        @i18n:param theme
        @i18n:type str
        @i18n:values ["light", "dark", "presentation"]
        @i18n:desc_es Nombre del tema a utilizar.
        @i18n:desc_en Theme name to use.
        @i18n:desc_fr Nom du thème à utiliser.
    **kwargs
        @i18n:param kwargs
        @i18n:desc_es Argumentos adicionales para VisualizationEngine.
        @i18n:desc_en Additional arguments for VisualizationEngine.
        @i18n:desc_fr Arguments supplémentaires pour VisualizationEngine.
    
    Returns
    -------
    VisualizationEngine
        @i18n:return engine
        @i18n:desc_es Motor de visualización configurado.
        @i18n:desc_en Configured visualization engine.
        @i18n:desc_fr Moteur de visualisation configuré.
    """
    theme_enum = Theme[theme.upper()]
    return VisualizationEngine(theme=theme_enum, **kwargs)


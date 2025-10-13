"""
Módulo de Visualización para Estructuras Topológicas

Este módulo proporciona funciones para visualizar complejos cúbicos y sus propiedades homológicas.

Autor: LatticeWeaver Team
Fecha: 13 de Octubre de 2025
"""

import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TopologyVisualizer:
    """
    Clase para visualizar complejos cúbicos y sus propiedades homológicas.
    """

    def __init__(self):
        """Inicializa el visualizador de topología."""
        self.fig = None
        self.ax = None

    def visualize_cubical_complex(
        self,
        cubical_complex,
        title: str = "Complejo Cúbico",
        figsize: tuple = (10, 8),
        node_color: str = "lightblue",
        edge_color: str = "gray",
        node_size: int = 500,
        font_size: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Visualiza un complejo cúbico como un grafo.

        Args:
            cubical_complex: Una instancia de CubicalComplex.
            title: Título del gráfico.
            figsize: Tamaño de la figura (ancho, alto).
            node_color: Color de los nodos.
            edge_color: Color de las aristas.
            node_size: Tamaño de los nodos.
            font_size: Tamaño de la fuente para las etiquetas.
            save_path: Ruta para guardar la figura (opcional).
        """
        if not hasattr(cubical_complex, 'graph'):
            raise ValueError("El objeto cubical_complex debe tener un atributo 'graph'.")

        self.fig, self.ax = plt.subplots(figsize=figsize)

        # Dibujar el grafo
        pos = nx.spring_layout(cubical_complex.graph, seed=42)
        nx.draw_networkx_nodes(
            cubical_complex.graph,
            pos,
            node_color=node_color,
            node_size=node_size,
            ax=self.ax
        )
        nx.draw_networkx_edges(
            cubical_complex.graph,
            pos,
            edge_color=edge_color,
            ax=self.ax
        )
        nx.draw_networkx_labels(
            cubical_complex.graph,
            pos,
            font_size=font_size,
            ax=self.ax
        )

        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Figura guardada en {save_path}")

        plt.tight_layout()
        return self.fig

    def visualize_homology(
        self,
        homology: Dict[str, int],
        title: str = "Números de Betti",
        figsize: tuple = (8, 6),
        bar_color: str = "steelblue",
        save_path: Optional[str] = None
    ):
        """
        Visualiza los números de Betti de un complejo cúbico como un gráfico de barras.

        Args:
            homology: Diccionario con los números de Betti (beta_0, beta_1, beta_2).
            title: Título del gráfico.
            figsize: Tamaño de la figura (ancho, alto).
            bar_color: Color de las barras.
            save_path: Ruta para guardar la figura (opcional).
        """
        if not isinstance(homology, dict):
            raise ValueError("El argumento 'homology' debe ser un diccionario.")

        self.fig, self.ax = plt.subplots(figsize=figsize)

        betti_numbers = [homology.get('beta_0', 0), homology.get('beta_1', 0), homology.get('beta_2', 0)]
        labels = ['β₀\n(Componentes)', 'β₁\n(Ciclos)', 'β₂\n(Cavidades)']

        bars = self.ax.bar(labels, betti_numbers, color=bar_color, alpha=0.7, edgecolor='black')

        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            self.ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )

        self.ax.set_ylabel('Número de Betti', fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.set_ylim(0, max(betti_numbers) + 1 if max(betti_numbers) > 0 else 1)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Figura guardada en {save_path}")

        plt.tight_layout()
        return self.fig

    def show(self):
        """Muestra la figura actual."""
        if self.fig:
            plt.show()
        else:
            logger.warning("No hay figura para mostrar.")

    def close(self):
        """Cierra la figura actual."""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


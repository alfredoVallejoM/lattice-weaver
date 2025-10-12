"""
Tests unitarios para el módulo de visualización.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path

from lattice_weaver.visualization import (
    load_trace,
    plot_search_tree,
    plot_domain_evolution,
    plot_backtrack_heatmap,
    generate_report
)


@pytest.fixture
def sample_trace_df():
    """Crea un DataFrame de ejemplo para testing."""
    import time
    
    events = [
        {
            'timestamp': time.time(),
            'event_type': 'search_started',
            'variable': None,
            'value': None,
            'source_variable': None,
            'pruned_values': None,
            'depth': 0,
            'metadata': '{}'
        },
        {
            'timestamp': time.time() + 0.001,
            'event_type': 'variable_assigned',
            'variable': 'x0',
            'value': 1,
            'source_variable': None,
            'pruned_values': None,
            'depth': 0,
            'metadata': '{}'
        },
        {
            'timestamp': time.time() + 0.002,
            'event_type': 'domain_pruned',
            'variable': 'x1',
            'value': None,
            'source_variable': 'x0',
            'pruned_values': '[1, 2]',
            'depth': 0,
            'metadata': '{}'
        },
        {
            'timestamp': time.time() + 0.003,
            'event_type': 'variable_assigned',
            'variable': 'x1',
            'value': 3,
            'source_variable': None,
            'pruned_values': None,
            'depth': 1,
            'metadata': '{}'
        },
        {
            'timestamp': time.time() + 0.004,
            'event_type': 'backtrack',
            'variable': 'x1',
            'value': None,
            'source_variable': None,
            'pruned_values': None,
            'depth': 1,
            'metadata': '{}'
        },
        {
            'timestamp': time.time() + 0.005,
            'event_type': 'solution_found',
            'variable': None,
            'value': None,
            'source_variable': None,
            'pruned_values': None,
            'depth': 1,
            'metadata': '{}'
        },
        {
            'timestamp': time.time() + 0.006,
            'event_type': 'search_ended',
            'variable': None,
            'value': None,
            'source_variable': None,
            'pruned_values': None,
            'depth': 0,
            'metadata': '{}'
        }
    ]
    
    return pd.DataFrame(events)


class TestLoadTrace:
    """Tests para la función load_trace."""
    
    def test_load_csv_trace(self):
        """Test de carga de trace en formato CSV."""
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('timestamp,event_type,variable,value,source_variable,pruned_values,depth,metadata\n')
            f.write('1234567890.0,search_started,,,,,0,{}\n')
            f.write('1234567890.1,variable_assigned,x0,1,,,0,{}\n')
            temp_path = f.name
        
        try:
            df = load_trace(temp_path)
            assert len(df) == 2
            assert 'event_type' in df.columns
            assert df['event_type'].iloc[0] == 'search_started'
        finally:
            Path(temp_path).unlink()
    
    def test_load_nonexistent_file(self):
        """Test de carga de archivo inexistente."""
        with pytest.raises(FileNotFoundError):
            load_trace('/nonexistent/path/trace.csv')


class TestPlotSearchTree:
    """Tests para plot_search_tree."""
    
    def test_plot_search_tree_with_data(self, sample_trace_df):
        """Test de generación del árbol de búsqueda con datos."""
        fig = plot_search_tree(sample_trace_df)
        
        assert fig is not None
        assert len(fig.data) > 0
        assert fig.layout.title.text == "Árbol de Búsqueda"
    
    def test_plot_search_tree_empty(self):
        """Test de generación del árbol con DataFrame vacío."""
        df = pd.DataFrame(columns=[
            'timestamp', 'event_type', 'variable', 'value',
            'source_variable', 'pruned_values', 'depth', 'metadata'
        ])
        
        fig = plot_search_tree(df)
        assert fig is not None
    
    def test_plot_search_tree_max_nodes(self, sample_trace_df):
        """Test de limitación de nodos máximos."""
        fig = plot_search_tree(sample_trace_df, max_nodes=2)
        assert fig is not None


class TestPlotDomainEvolution:
    """Tests para plot_domain_evolution."""
    
    def test_plot_domain_evolution_with_data(self, sample_trace_df):
        """Test de generación de evolución de dominios con datos."""
        fig = plot_domain_evolution(sample_trace_df)
        
        assert fig is not None
        assert fig.layout.title.text == 'Evolución de Podas de Dominio'
    
    def test_plot_domain_evolution_no_prunes(self):
        """Test de generación sin eventos de poda."""
        df = pd.DataFrame([{
            'timestamp': 1234567890.0,
            'event_type': 'search_started',
            'variable': None,
            'value': None,
            'source_variable': None,
            'pruned_values': None,
            'depth': 0,
            'metadata': '{}'
        }])
        
        fig = plot_domain_evolution(df)
        assert fig is not None


class TestPlotBacktrackHeatmap:
    """Tests para plot_backtrack_heatmap."""
    
    def test_plot_backtrack_heatmap_with_data(self, sample_trace_df):
        """Test de generación de heatmap con datos."""
        fig = plot_backtrack_heatmap(sample_trace_df)
        
        assert fig is not None
        assert fig.layout.title.text == 'Heatmap de Backtracks'
    
    def test_plot_backtrack_heatmap_no_backtracks(self):
        """Test de generación sin backtracks."""
        df = pd.DataFrame([{
            'timestamp': 1234567890.0,
            'event_type': 'search_started',
            'variable': None,
            'value': None,
            'source_variable': None,
            'pruned_values': None,
            'depth': 0,
            'metadata': '{}'
        }])
        
        fig = plot_backtrack_heatmap(df)
        assert fig is not None


class TestGenerateReport:
    """Tests para generate_report."""
    
    def test_generate_report(self, sample_trace_df):
        """Test de generación de reporte completo."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name
        
        try:
            generate_report(sample_trace_df, output_path, title="Test Report")
            
            # Verificar que el archivo se creó
            assert Path(output_path).exists()
            
            # Verificar contenido básico
            content = Path(output_path).read_text()
            assert "Test Report" in content
            assert "Resumen de Estadísticas" in content
            assert "Árbol de Búsqueda" in content
            
        finally:
            Path(output_path).unlink()
    
    def test_generate_report_creates_directory(self, sample_trace_df):
        """Test de creación de directorio si no existe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "report.html"
            
            generate_report(sample_trace_df, str(output_path))
            
            assert output_path.exists()


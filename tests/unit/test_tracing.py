"""
Tests unitarios para el módulo de tracing.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import pytest
import time
import tempfile
import json
from pathlib import Path

from lattice_weaver.arc_weaver.tracing import (
    SearchEvent,
    SearchSpaceTracer,
    load_trace,
    load_trace_csv,
    load_trace_jsonl
)


class TestSearchEvent:
    """Tests para la clase SearchEvent."""
    
    def test_create_basic_event(self):
        """Test de creación básica de evento."""
        event = SearchEvent(
            timestamp=time.time(),
            event_type='variable_assigned',
            variable='x1',
            value=3,
            depth=1
        )
        
        assert event.event_type == 'variable_assigned'
        assert event.variable == 'x1'
        assert event.value == 3
        assert event.depth == 1
    
    def test_event_to_dict(self):
        """Test de conversión a diccionario."""
        event = SearchEvent(
            timestamp=123.456,
            event_type='backtrack',
            variable='x2',
            depth=2
        )
        
        d = event.to_dict()
        
        assert d['timestamp'] == 123.456
        assert d['event_type'] == 'backtrack'
        assert d['variable'] == 'x2'
        assert d['depth'] == 2
    
    def test_event_with_pruned_values(self):
        """Test de evento con valores podados."""
        event = SearchEvent(
            timestamp=time.time(),
            event_type='domain_pruned',
            variable='x1',
            source_variable='x2',
            pruned_values={1, 2, 3},
            depth=1
        )
        
        d = event.to_dict()
        
        # pruned_values debe convertirse a lista
        assert isinstance(d['pruned_values'], list)
        assert set(d['pruned_values']) == {1, 2, 3}


class TestSearchSpaceTracer:
    """Tests para la clase SearchSpaceTracer."""
    
    def test_tracer_disabled(self):
        """Test de tracer deshabilitado."""
        tracer = SearchSpaceTracer(enabled=False)
        
        tracer.start()
        tracer.record(SearchEvent(
            timestamp=time.time(),
            event_type='search_started'
        ))
        tracer.stop()
        
        # No debe haber eventos
        assert len(tracer.events) == 0
    
    def test_tracer_enabled_memory_only(self):
        """Test de tracer habilitado solo en memoria."""
        tracer = SearchSpaceTracer(enabled=True)
        
        tracer.start()
        
        # Registrar varios eventos
        tracer.record(SearchEvent(
            timestamp=time.time(),
            event_type='search_started'
        ))
        
        tracer.record(SearchEvent(
            timestamp=time.time(),
            event_type='variable_assigned',
            variable='x1',
            value=1,
            depth=0
        ))
        
        tracer.record(SearchEvent(
            timestamp=time.time(),
            event_type='backtrack',
            variable='x1',
            depth=0
        ))
        
        tracer.record(SearchEvent(
            timestamp=time.time(),
            event_type='search_ended'
        ))
        
        tracer.stop()
        
        # Verificar eventos
        assert len(tracer.events) == 4
        assert tracer.events[0].event_type == 'search_started'
        assert tracer.events[1].event_type == 'variable_assigned'
        assert tracer.events[2].event_type == 'backtrack'
        assert tracer.events[3].event_type == 'search_ended'
    
    def test_tracer_csv_output(self):
        """Test de salida a CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            tracer = SearchSpaceTracer(
                enabled=True,
                output_path=output_path,
                output_format='csv'
            )
            
            tracer.start()
            
            tracer.record(SearchEvent(
                timestamp=time.time(),
                event_type='search_started'
            ))
            
            tracer.record(SearchEvent(
                timestamp=time.time(),
                event_type='variable_assigned',
                variable='x1',
                value=1,
                depth=0
            ))
            
            tracer.stop()
            
            # Verificar que el archivo existe
            assert Path(output_path).exists()
            
            # Cargar y verificar
            df = load_trace_csv(output_path)
            assert len(df) == 2
            assert df.iloc[0]['event_type'] == 'search_started'
            assert df.iloc[1]['event_type'] == 'variable_assigned'
            
        finally:
            Path(output_path).unlink()
    
    def test_tracer_jsonl_output(self):
        """Test de salida a JSON Lines."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            output_path = f.name
        
        try:
            tracer = SearchSpaceTracer(
                enabled=True,
                output_path=output_path,
                output_format='jsonl'
            )
            
            tracer.start()
            
            tracer.record(SearchEvent(
                timestamp=time.time(),
                event_type='search_started'
            ))
            
            tracer.record(SearchEvent(
                timestamp=time.time(),
                event_type='variable_assigned',
                variable='x1',
                value=1,
                depth=0
            ))
            
            tracer.stop()
            
            # Verificar que el archivo existe
            assert Path(output_path).exists()
            
            # Cargar y verificar
            df = load_trace_jsonl(output_path)
            assert len(df) == 2
            assert df.iloc[0]['event_type'] == 'search_started'
            assert df.iloc[1]['event_type'] == 'variable_assigned'
            
        finally:
            Path(output_path).unlink()
    
    def test_statistics_calculation(self):
        """Test de cálculo de estadísticas."""
        tracer = SearchSpaceTracer(enabled=True)
        
        tracer.start()
        
        # Simular una búsqueda
        tracer.record(SearchEvent(
            timestamp=time.time(),
            event_type='search_started'
        ))
        
        # 3 asignaciones
        for i in range(3):
            tracer.record(SearchEvent(
                timestamp=time.time(),
                event_type='variable_assigned',
                variable=f'x{i}',
                value=i,
                depth=i
            ))
        
        # 2 backtracks
        for i in range(2):
            tracer.record(SearchEvent(
                timestamp=time.time(),
                event_type='backtrack',
                variable=f'x{i}',
                depth=i
            ))
        
        # 1 solución
        tracer.record(SearchEvent(
            timestamp=time.time(),
            event_type='solution_found'
        ))
        
        # 2 llamadas AC-3
        for i in range(2):
            tracer.record(SearchEvent(
                timestamp=time.time(),
                event_type='ac3_call'
            ))
        
        tracer.record(SearchEvent(
            timestamp=time.time(),
            event_type='search_ended'
        ))
        
        tracer.stop()
        
        stats = tracer.get_statistics()
        
        assert stats['nodes_explored'] == 3
        assert stats['backtracks'] == 2
        assert stats['solutions_found'] == 1
        assert stats['ac3_calls'] == 2
        assert stats['max_depth'] == 2
        assert stats['backtrack_rate'] == pytest.approx(2/3, rel=1e-6)
        # Total: 1 search_started + 3 variable_assigned + 2 backtrack + 1 solution_found + 2 ac3_call + 1 search_ended = 10
        assert stats['total_events'] == 10
    
    def test_context_manager(self):
        """Test del context manager."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            with SearchSpaceTracer(enabled=True, output_path=output_path) as tracer:
                tracer.record(SearchEvent(
                    timestamp=time.time(),
                    event_type='search_started'
                ))
                
                tracer.record(SearchEvent(
                    timestamp=time.time(),
                    event_type='search_ended'
                ))
            
            # Verificar que el archivo se creó
            assert Path(output_path).exists()
            
            # Verificar contenido
            df = load_trace(output_path)
            assert len(df) == 2
            
        finally:
            Path(output_path).unlink()
    
    def test_buffer_flush(self):
        """Test de vaciado del buffer."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            tracer = SearchSpaceTracer(
                enabled=True,
                output_path=output_path,
                buffer_size=5  # Buffer pequeño para forzar flush
            )
            
            tracer.start()
            
            # Registrar más eventos que el tamaño del buffer
            for i in range(10):
                tracer.record(SearchEvent(
                    timestamp=time.time(),
                    event_type='variable_assigned',
                    variable=f'x{i}',
                    value=i,
                    depth=i
                ))
            
            tracer.stop()
            
            # Verificar que todos los eventos se escribieron
            df = load_trace(output_path)
            assert len(df) == 10
            
        finally:
            Path(output_path).unlink()
    
    def test_to_dataframe(self):
        """Test de conversión a DataFrame."""
        tracer = SearchSpaceTracer(enabled=True)
        
        tracer.start()
        
        tracer.record(SearchEvent(
            timestamp=time.time(),
            event_type='search_started'
        ))
        
        tracer.record(SearchEvent(
            timestamp=time.time(),
            event_type='variable_assigned',
            variable='x1',
            value=1,
            depth=0
        ))
        
        tracer.stop()
        
        df = tracer.to_dataframe()
        
        assert len(df) == 2
        assert 'event_type' in df.columns
        assert 'variable' in df.columns
        assert 'depth' in df.columns
        assert df.iloc[0]['event_type'] == 'search_started'
        assert df.iloc[1]['variable'] == 'x1'
    
    def test_clear(self):
        """Test de limpieza de eventos."""
        tracer = SearchSpaceTracer(enabled=True)
        
        tracer.start()
        
        tracer.record(SearchEvent(
            timestamp=time.time(),
            event_type='search_started'
        ))
        
        tracer.record(SearchEvent(
            timestamp=time.time(),
            event_type='search_ended'
        ))
        
        tracer.stop()
        
        assert len(tracer.events) == 2
        
        tracer.clear()
        
        assert len(tracer.events) == 0
        assert tracer.get_statistics()['nodes_explored'] == 0


class TestLoadTraceFunctions:
    """Tests para las funciones de carga de traces."""
    
    def test_load_trace_auto_detect_csv(self):
        """Test de detección automática de formato CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            tracer = SearchSpaceTracer(enabled=True, output_path=output_path)
            tracer.start()
            tracer.record(SearchEvent(timestamp=time.time(), event_type='search_started'))
            tracer.stop()
            
            df = load_trace(output_path)
            assert len(df) == 1
            
        finally:
            Path(output_path).unlink()
    
    def test_load_trace_auto_detect_jsonl(self):
        """Test de detección automática de formato JSONL."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            output_path = f.name
        
        try:
            tracer = SearchSpaceTracer(
                enabled=True,
                output_path=output_path,
                output_format='jsonl'
            )
            tracer.start()
            tracer.record(SearchEvent(timestamp=time.time(), event_type='search_started'))
            tracer.stop()
            
            df = load_trace(output_path)
            assert len(df) == 1
            
        finally:
            Path(output_path).unlink()
    
    def test_load_trace_unsupported_format(self):
        """Test de formato no soportado."""
        with pytest.raises(ValueError, match="Formato de archivo no soportado"):
            load_trace("test.txt")


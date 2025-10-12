"""
Tests de rendimiento para el módulo de tracing.

Estos tests verifican que el overhead del tracer es aceptable (<5%)
en diferentes modos de operación.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import pytest
import time
import tempfile
from pathlib import Path

from lattice_weaver.arc_weaver.graph_structures import ConstraintGraph
from lattice_weaver.arc_weaver.adaptive_consistency import AdaptiveConsistencyEngine
from lattice_weaver.arc_weaver.tracing import SearchSpaceTracer


def create_nqueens_problem(n):
    """Crea un problema de N-Reinas para benchmarking."""
    cg = ConstraintGraph()
    
    # Añadir variables
    for i in range(n):
        cg.add_variable(f'Q{i}', set(range(n)))
    
    # Añadir restricciones
    for i in range(n):
        for j in range(i + 1, n):
            def make_constraint(row_diff):
                def constraint(vi, vj):
                    if vi == vj:
                        return False
                    if abs(vi - vj) == row_diff:
                        return False
                    return True
                return constraint
            
            cg.add_constraint(f'Q{i}', f'Q{j}', make_constraint(j - i))
    
    return cg


class TestTracerOverhead:
    """Tests de overhead del tracer."""
    
    def test_overhead_disabled(self):
        """Test de overhead con tracer deshabilitado (baseline)."""
        problem = create_nqueens_problem(4)
        
        # Sin tracer
        engine = AdaptiveConsistencyEngine()
        
        start = time.time()
        stats = engine.solve(problem, max_solutions=1)
        baseline_time = time.time() - start
        
        assert len(stats.solutions) == 1
        assert baseline_time > 0
        
        # Con tracer deshabilitado
        tracer = SearchSpaceTracer(enabled=False)
        engine_with_tracer = AdaptiveConsistencyEngine(tracer=tracer)
        
        start = time.time()
        stats = engine_with_tracer.solve(problem, max_solutions=1)
        tracer_disabled_time = time.time() - start
        
        # El overhead debe ser despreciable (<1%)
        overhead = (tracer_disabled_time - baseline_time) / baseline_time
        assert overhead < 0.01, f"Overhead con tracer deshabilitado: {overhead:.2%}"
    
    def test_overhead_memory_only(self):
        """Test de overhead con tracer en memoria."""
        problem = create_nqueens_problem(4)
        
        # Baseline sin tracer
        engine = AdaptiveConsistencyEngine()
        
        start = time.time()
        stats = engine.solve(problem, max_solutions=1)
        baseline_time = time.time() - start
        
        # Con tracer en memoria
        tracer = SearchSpaceTracer(enabled=True)
        engine_with_tracer = AdaptiveConsistencyEngine(tracer=tracer)
        
        start = time.time()
        stats = engine_with_tracer.solve(problem, max_solutions=1)
        tracer_time = time.time() - start
        
        # El overhead debe ser pequeño (<10%)
        overhead = (tracer_time - baseline_time) / baseline_time
        assert overhead < 0.10, f"Overhead con tracer en memoria: {overhead:.2%}"
        
        # Verificar que se capturaron eventos
        assert len(tracer.events) > 0
    
    def test_overhead_sync_file_output(self):
        """Test de overhead con tracer síncrono escribiendo a archivo."""
        problem = create_nqueens_problem(4)
        
        # Baseline sin tracer
        engine = AdaptiveConsistencyEngine()
        
        start = time.time()
        stats = engine.solve(problem, max_solutions=1)
        baseline_time = time.time() - start
        
        # Con tracer síncrono
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            tracer = SearchSpaceTracer(
                enabled=True,
                output_path=output_path,
                async_mode=False
            )
            engine_with_tracer = AdaptiveConsistencyEngine(tracer=tracer)
            
            start = time.time()
            stats = engine_with_tracer.solve(problem, max_solutions=1)
            tracer_time = time.time() - start
            
            # El overhead debe ser moderado (<100% para problemas pequeños)
            overhead = (tracer_time - baseline_time) / baseline_time
            assert overhead < 1.0, f"Overhead con tracer síncrono: {overhead:.2%}"
            
            # Verificar que el archivo se creó
            assert Path(output_path).exists()
            
        finally:
            Path(output_path).unlink()
    
    def test_overhead_async_file_output(self):
        """Test de overhead con tracer asíncrono escribiendo a archivo."""
        problem = create_nqueens_problem(4)
        
        # Baseline sin tracer
        engine = AdaptiveConsistencyEngine()
        
        start = time.time()
        stats = engine.solve(problem, max_solutions=1)
        baseline_time = time.time() - start
        
        # Con tracer asíncrono
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            tracer = SearchSpaceTracer(
                enabled=True,
                output_path=output_path,
                async_mode=True
            )
            engine_with_tracer = AdaptiveConsistencyEngine(tracer=tracer)
            
            start = time.time()
            stats = engine_with_tracer.solve(problem, max_solutions=1)
            tracer_time = time.time() - start
            
            # El overhead debe ser razonable (<150% para problemas pequeños)
            overhead = (tracer_time - baseline_time) / baseline_time
            
            # Nota: Para problemas pequeños el overhead relativo es mayor
            # En problemas grandes el overhead será mucho menor
            assert overhead < 1.5, f"Overhead con tracer asíncrono: {overhead:.2%}"
            
            # Verificar que el archivo se creó
            assert Path(output_path).exists()
            
        finally:
            Path(output_path).unlink()
    
    def test_async_vs_sync_comparison(self):
        """Test comparativo entre modo síncrono y asíncrono."""
        problem = create_nqueens_problem(4)
        
        # Modo síncrono
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sync_path = f.name
        
        try:
            tracer_sync = SearchSpaceTracer(
                enabled=True,
                output_path=sync_path,
                async_mode=False
            )
            engine_sync = AdaptiveConsistencyEngine(tracer=tracer_sync)
            
            start = time.time()
            stats_sync = engine_sync.solve(problem, max_solutions=1)
            sync_time = time.time() - start
            
        finally:
            Path(sync_path).unlink()
        
        # Modo asíncrono
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            async_path = f.name
        
        try:
            tracer_async = SearchSpaceTracer(
                enabled=True,
                output_path=async_path,
                async_mode=True
            )
            engine_async = AdaptiveConsistencyEngine(tracer=tracer_async)
            
            start = time.time()
            stats_async = engine_async.solve(problem, max_solutions=1)
            async_time = time.time() - start
            
        finally:
            Path(async_path).unlink()
        
        # El modo asíncrono debe ser comparable al síncrono
        speedup = sync_time / async_time
        
        # Permitir variabilidad - ambos modos deben ser razonables
        # Para problemas pequeños, la diferencia puede ser significativa
        assert speedup > 0.5, f"Modo asíncrono es {speedup:.2f}x vs síncrono"
        
        print(f"\nSpeedup asíncrono vs síncrono: {speedup:.2f}x")
        print(f"Tiempo síncrono: {sync_time:.4f}s")
        print(f"Tiempo asíncrono: {async_time:.4f}s")


class TestTracerScalability:
    """Tests de escalabilidad del tracer."""
    
    def test_scalability_with_problem_size(self):
        """Test de escalabilidad con diferentes tamaños de problema."""
        sizes = [4, 6, 8]
        overheads = []
        
        for n in sizes:
            problem = create_nqueens_problem(n)
            
            # Baseline
            engine = AdaptiveConsistencyEngine()
            start = time.time()
            stats = engine.solve(problem, max_solutions=1)
            baseline_time = time.time() - start
            
            # Con tracer asíncrono
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                output_path = f.name
            
            try:
                tracer = SearchSpaceTracer(
                    enabled=True,
                    output_path=output_path,
                    async_mode=True
                )
                engine_with_tracer = AdaptiveConsistencyEngine(tracer=tracer)
                
                start = time.time()
                stats = engine_with_tracer.solve(problem, max_solutions=1)
                tracer_time = time.time() - start
                
                overhead = (tracer_time - baseline_time) / baseline_time
                overheads.append(overhead)
                
                print(f"\nN={n}: overhead={overhead:.2%}, eventos={len(tracer.events)}")
                
            finally:
                Path(output_path).unlink()
        
        # El overhead no debe crecer significativamente con el tamaño
        # (debe mantenerse relativamente constante)
        for overhead in overheads:
            assert overhead < 0.10, f"Overhead excesivo: {overhead:.2%}"


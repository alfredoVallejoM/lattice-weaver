"""
Tests para Utilidades de Fase 1 (Quick Wins)

Tests para:
- Sparse Set
- Object Pool
- Auto Profiler
- Lazy Initialization

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import pytest
from lattice_weaver.utils.sparse_set import SparseSet, create_sparse_set, sparse_set_from_list
from lattice_weaver.utils.object_pool import (
    ObjectPool, ObjectPoolWithManagement,
    get_list_pool, get_dict_pool, get_set_pool,
    pooled_list, pooled_dict, pooled_set
)
from lattice_weaver.utils.auto_profiler import AutoProfiler, OptimizationLevel
from lattice_weaver.utils.lazy_init import LazyProperty, LazyObject, lazy_init, lazy_method, ConditionalInit


class TestSparseSet:
    """Tests para Sparse Set."""
    
    def test_initialization(self):
        """Test inicialización básica."""
        ss = SparseSet([1, 2, 3, 4, 5])
        assert len(ss) == 5
        assert list(ss) == [1, 2, 3, 4, 5]
    
    def test_contains(self):
        """Test verificación de pertenencia."""
        ss = SparseSet([1, 2, 3])
        assert 1 in ss
        assert 2 in ss
        assert 3 in ss
        assert 4 not in ss
    
    def test_remove(self):
        """Test eliminación de valores."""
        ss = SparseSet([1, 2, 3, 4, 5])
        
        assert ss.remove(3)
        assert len(ss) == 4
        assert 3 not in ss
        assert list(ss) == [1, 2, 5, 4]  # Orden puede cambiar
        
        # Eliminar de nuevo no hace nada
        assert not ss.remove(3)
        assert len(ss) == 4
    
    def test_add(self):
        """Test añadir valores."""
        ss = SparseSet([1, 2, 3])
        ss.remove(2)
        
        assert ss.add(2)
        assert len(ss) == 3
        assert 2 in ss
        
        # Añadir de nuevo no hace nada
        assert not ss.add(2)
        assert len(ss) == 3
    
    def test_clear_and_reset(self):
        """Test clear y reset."""
        ss = SparseSet([1, 2, 3, 4, 5])
        
        ss.clear()
        assert len(ss) == 0
        assert list(ss) == []
        
        ss.reset()
        assert len(ss) == 5
        assert set(ss) == {1, 2, 3, 4, 5}
    
    def test_snapshot_restore(self):
        """Test snapshot y restore para backtracking."""
        ss = SparseSet([1, 2, 3, 4, 5])
        
        # Estado inicial
        snapshot1 = ss.snapshot()
        
        # Modificar
        ss.remove(3)
        ss.remove(4)
        assert len(ss) == 3
        
        # Snapshot intermedio
        snapshot2 = ss.snapshot()
        
        # Modificar más
        ss.remove(5)
        assert len(ss) == 2
        
        # Restaurar a snapshot2
        ss.restore(snapshot2)
        assert len(ss) == 3
        assert 3 not in ss
        assert 5 in ss
        
        # Restaurar a snapshot1
        ss.restore(snapshot1)
        assert len(ss) == 5
        assert 3 in ss
    
    def test_copy(self):
        """Test copia de Sparse Set."""
        ss1 = SparseSet([1, 2, 3])
        ss1.remove(2)
        
        ss2 = ss1.copy()
        assert len(ss2) == 2
        assert 2 not in ss2
        
        # Modificar copia no afecta original
        ss2.add(2)
        assert 2 in ss2
        assert 2 not in ss1
    
    def test_factory_functions(self):
        """Test funciones factory."""
        ss1 = create_sparse_set([1, 2, 3])
        assert len(ss1) == 3
        
        ss2 = sparse_set_from_list([1, 2, 3, 4, 5], [1, 3, 5])
        assert len(ss2) == 3
        assert 1 in ss2
        assert 2 not in ss2
        assert 3 in ss2


class TestObjectPool:
    """Tests para Object Pool."""
    
    def test_basic_pool(self):
        """Test pool básico."""
        pool = ObjectPool(factory=list, reset=lambda lst: lst.clear())
        
        # Adquirir crea nuevo objeto
        lst1 = pool.acquire()
        assert isinstance(lst1, list)
        assert pool.get_stats()['created'] == 1
        
        # Liberar y adquirir reutiliza
        pool.release(lst1)
        lst2 = pool.acquire()
        assert lst2 is lst1
        assert pool.get_stats()['reused'] == 1
    
    def test_pool_max_size(self):
        """Test tamaño máximo del pool."""
        pool = ObjectPool(factory=list, max_size=2)
        
        lst1 = pool.acquire()
        lst2 = pool.acquire()
        lst3 = pool.acquire()
        
        pool.release(lst1)
        pool.release(lst2)
        pool.release(lst3)  # Este se descarta
        
        assert pool.get_stats()['discarded'] == 1
    
    def test_pool_reset(self):
        """Test reset de objetos."""
        pool = ObjectPool(factory=list, reset=lambda lst: lst.clear())
        
        lst = pool.acquire()
        lst.extend([1, 2, 3])
        pool.release(lst)
        
        lst2 = pool.acquire()
        assert lst2 is lst
        assert len(lst2) == 0  # Fue reseteado
    
    def test_pool_with_management(self):
        """Test pool con context manager."""
        pool = ObjectPoolWithManagement(factory=list, reset=lambda lst: lst.clear())
        
        with pool.acquire_managed() as lst:
            lst.extend([1, 2, 3])
        
        # Objeto fue liberado automáticamente
        assert pool.get_stats()['released'] == 1
    
    def test_global_pools(self):
        """Test pools globales."""
        # List pool
        lst = get_list_pool().acquire()
        assert isinstance(lst, list)
        get_list_pool().release(lst)
        
        # Dict pool
        d = get_dict_pool().acquire()
        assert isinstance(d, dict)
        get_dict_pool().release(d)
        
        # Set pool
        s = get_set_pool().acquire()
        assert isinstance(s, set)
        get_set_pool().release(s)
    
    def test_context_managers(self):
        """Test context managers de pools globales."""
        with pooled_list() as lst:
            lst.extend([1, 2, 3])
            assert len(lst) == 3
        
        with pooled_dict() as d:
            d['key'] = 'value'
            assert len(d) == 1
        
        with pooled_set() as s:
            s.add(1)
            assert len(s) == 1


class TestAutoProfiler:
    """Tests para Auto Profiler."""
    
    def test_initialization(self):
        """Test inicialización."""
        profiler = AutoProfiler(profiling_backtracks=50)
        assert profiler.profiling_backtracks == 50
        assert profiler.metrics.num_backtracks == 0
    
    def test_record_problem_characteristics(self):
        """Test registro de características del problema."""
        profiler = AutoProfiler()
        profiler.record_problem_characteristics(
            num_variables=10,
            num_constraints=20,
            domain_sizes=[5, 10, 15],
            has_soft=True,
            has_hierarchy=True
        )
        
        assert profiler.metrics.num_variables == 10
        assert profiler.metrics.num_constraints == 20
        assert profiler.metrics.avg_domain_size == 10.0
        assert profiler.metrics.max_domain_size == 15
        assert profiler.metrics.has_soft_constraints == True
        assert profiler.metrics.has_hierarchy == True
    
    def test_record_metrics(self):
        """Test registro de métricas."""
        profiler = AutoProfiler()
        
        profiler.record_backtrack()
        assert profiler.metrics.num_backtracks == 1
        
        profiler.record_propagation(5)
        assert profiler.metrics.num_propagations == 1
        assert profiler.metrics.propagation_effectiveness == 5.0
        
        profiler.record_constraint_evaluation()
        assert profiler.metrics.num_constraint_evaluations == 1
        
        profiler.record_energy_computation()
        assert profiler.metrics.num_energy_computations == 1
    
    def test_should_continue_profiling(self):
        """Test decisión de continuar profiling."""
        profiler = AutoProfiler(profiling_backtracks=10)
        
        assert profiler.should_continue_profiling() == True
        
        for _ in range(10):
            profiler.record_backtrack()
        
        assert profiler.should_continue_profiling() == False
    
    def test_analyze_small_problem(self):
        """Test análisis de problema pequeño."""
        profiler = AutoProfiler()
        profiler.record_problem_characteristics(
            num_variables=10,
            num_constraints=15,
            domain_sizes=[5, 5, 5],
            has_soft=False,
            has_hierarchy=False
        )
        
        rec = profiler.analyze_and_recommend()
        
        assert rec.optimization_level == OptimizationLevel.LITE
        assert rec.use_tms == False
        assert rec.use_homotopy_rules == False
    
    def test_analyze_large_problem(self):
        """Test análisis de problema grande."""
        profiler = AutoProfiler()
        profiler.record_problem_characteristics(
            num_variables=150,
            num_constraints=500,
            domain_sizes=[100] * 150,
            has_soft=True,
            has_hierarchy=True
        )
        
        rec = profiler.analyze_and_recommend()
        
        # El profiler detecta MEDIUM porque tiene SOFT (no es solo grande)
        assert rec.optimization_level in [OptimizationLevel.MEDIUM, OptimizationLevel.FULL]
        assert rec.use_tms == True
        assert rec.use_homotopy_rules == True
        assert rec.use_sparse_sets == True


class TestLazyInit:
    """Tests para Lazy Initialization."""
    
    def test_lazy_property(self):
        """Test LazyProperty."""
        class MyClass:
            def __init__(self):
                self.init_count = 0
            
            @LazyProperty
            def expensive_property(self):
                self.init_count += 1
                return "expensive_value"
        
        obj = MyClass()
        assert obj.init_count == 0
        
        # Primera acceso inicializa
        value1 = obj.expensive_property
        assert value1 == "expensive_value"
        assert obj.init_count == 1
        
        # Segunda acceso no reinicializa
        value2 = obj.expensive_property
        assert value2 == "expensive_value"
        assert obj.init_count == 1
    
    def test_lazy_object(self):
        """Test LazyObject."""
        call_count = [0]
        
        def factory():
            call_count[0] += 1
            return "value"
        
        lazy_obj = LazyObject(factory)
        
        assert not lazy_obj.is_initialized()
        assert call_count[0] == 0
        
        # Primera acceso inicializa
        value1 = lazy_obj.get()
        assert value1 == "value"
        assert lazy_obj.is_initialized()
        assert call_count[0] == 1
        
        # Segunda acceso no reinicializa
        value2 = lazy_obj.get()
        assert value2 == "value"
        assert call_count[0] == 1
        
        # Reset
        lazy_obj.reset()
        assert not lazy_obj.is_initialized()
    
    def test_lazy_init_function(self):
        """Test función lazy_init."""
        lazy_obj = lazy_init(lambda: [1, 2, 3])
        
        assert not lazy_obj.is_initialized()
        
        value = lazy_obj.get()
        assert value == [1, 2, 3]
        assert lazy_obj.is_initialized()
    
    def test_lazy_method(self):
        """Test decorator lazy_method."""
        class MyClass:
            def __init__(self):
                self.call_count = 0
            
            @lazy_method
            def expensive_method(self):
                self.call_count += 1
                return "result"
        
        obj = MyClass()
        
        # Primera llamada ejecuta
        result1 = obj.expensive_method()
        assert result1 == "result"
        assert obj.call_count == 1
        
        # Segunda llamada usa caché
        result2 = obj.expensive_method()
        assert result2 == "result"
        assert obj.call_count == 1
    
    def test_conditional_init(self):
        """Test ConditionalInit."""
        cond_init = ConditionalInit()
        
        # Registrar componentes
        cond_init.register("comp1", lambda: "value1", enabled=True)
        cond_init.register("comp2", lambda: "value2", enabled=False)
        
        # Componente habilitado se inicializa
        value1 = cond_init.get("comp1")
        assert value1 == "value1"
        assert cond_init.is_initialized("comp1")
        
        # Componente deshabilitado retorna None
        value2 = cond_init.get("comp2")
        assert value2 is None
        assert not cond_init.is_initialized("comp2")
        
        # Habilitar componente
        cond_init.enable("comp2")
        value2 = cond_init.get("comp2")
        assert value2 == "value2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


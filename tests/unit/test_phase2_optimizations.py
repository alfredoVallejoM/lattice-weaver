"""
Tests para Fase 2: Core Optimizations

Tests para:
- Adaptive Propagation
- Watched Literals
- Advanced Heuristics
- Predicate Cache

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import pytest
from lattice_weaver.arc_engine.adaptive_propagation import (
    AdaptivePropagationEngine, PropagationLevel
)
from lattice_weaver.fibration.watched_literals import (
    WatchedLiteralsManager, WatchedLiteralsConstraintChecker
)
from lattice_weaver.fibration.advanced_heuristics import (
    WeightedDegreeHeuristic, ImpactBasedSearch, ConflictDirectedVariableOrdering
)
from lattice_weaver.fibration.predicate_cache import (
    PredicateCache, CachedPredicate, PredicateCacheManager
)
from lattice_weaver.fibration.general_constraint import GeneralConstraint as Constraint
from lattice_weaver.arc_engine.core import ArcEngine


class TestAdaptivePropagation:
    """Tests para Adaptive Propagation."""
    
    def test_initialization(self):
        """Test inicialización."""
        arc_engine = ArcEngine()
        engine = AdaptivePropagationEngine(arc_engine)
        
        assert engine.current_level == PropagationLevel.AC3
        assert engine.propagation_count == 0
    
    def test_level_adaptation(self):
        """Test adaptación de nivel."""
        arc_engine = ArcEngine()
        engine = AdaptivePropagationEngine(
            arc_engine,
            initial_level=PropagationLevel.AC3,
            adaptation_threshold=10
        )
        
        # Simular propagaciones inefectivas
        for _ in range(10):
            engine.stats[PropagationLevel.AC3].num_propagations += 1
            engine.stats[PropagationLevel.AC3].values_eliminated += 0
            engine.stats[PropagationLevel.AC3].time_spent += 1.0
            engine.propagation_count += 1
        
        # Adaptar
        engine._adapt_level()
        
        # Debería bajar de AC3 a FC
        assert engine.current_level == PropagationLevel.FC


class TestWatchedLiterals:
    """Tests para Watched Literals."""
    
    def test_initialization(self):
        """Test inicialización."""
        constraints = [
            Constraint(
                variables=["A", "B"],
                predicate=lambda a: a["A"] != a["B"]
            ),
            Constraint(
                variables=["B", "C"],
                predicate=lambda a: a["B"] != a["C"]
            )
        ]
        
        manager = WatchedLiteralsManager(constraints)
        
        assert len(manager.watched) == 2
        assert len(manager.var_to_constraints) > 0
    
    def test_notify_assignment(self):
        """Test notificación de asignación."""
        constraints = [
            Constraint(
                variables=["A", "B", "C"],
                predicate=lambda a: a.get("A", 0) + a.get("B", 0) + a.get("C", 0) < 10
            )
        ]
        
        manager = WatchedLiteralsManager(constraints)
        
        # Asignar variable
        assignment = {"A": 1}
        to_check = manager.notify_assignment("A", 1, assignment)
        
        # Debería retornar restricciones a revisar (o vacío si encuentra nueva watched)
        assert isinstance(to_check, list)
    
    def test_constraint_checker(self):
        """Test checker con watched literals."""
        constraints = [
            Constraint(
                variables=["A", "B"],
                predicate=lambda a: a.get("A", 0) != a.get("B", 0)
            )
        ]
        
        checker = WatchedLiteralsConstraintChecker(constraints)
        
        # Asignación válida
        assignment = {"A": 1, "B": 2}
        satisfied, violated = checker.check_constraints(assignment, "A")
        
        assert satisfied == True
        assert len(violated) == 0


class TestWeightedDegreeHeuristic:
    """Tests para Weighted Degree Heuristic."""
    
    def test_initialization(self):
        """Test inicialización."""
        constraints = [
            Constraint(
                variables=["A", "B"],
                predicate=lambda a: True
            ),
            Constraint(
                variables=["B", "C"],
                predicate=lambda a: True
            )
        ]
        
        wdeg = WeightedDegreeHeuristic(constraints)
        
        assert len(wdeg.weights) == 2
        assert all(w == 1.0 for w in wdeg.weights.values())
    
    def test_record_conflict(self):
        """Test registro de conflictos."""
        constraint = Constraint(
            variables=["A", "B"],
            predicate=lambda a: True
        )
        
        wdeg = WeightedDegreeHeuristic([constraint])
        
        initial_weight = wdeg.weights[id(constraint)]
        wdeg.record_conflict(constraint)
        
        assert wdeg.weights[id(constraint)] == initial_weight + 1.0
        assert wdeg.conflict_count == 1
    
    def test_get_weighted_degree(self):
        """Test cálculo de weighted degree."""
        constraints = [
            Constraint(variables=["A", "B"], predicate=lambda a: True),
            Constraint(variables=["A", "C"], predicate=lambda a: True)
        ]
        
        wdeg = WeightedDegreeHeuristic(constraints)
        
        # A está en 2 restricciones
        degree_a = wdeg.get_weighted_degree("A")
        assert degree_a == 2.0
        
        # B está en 1 restricción
        degree_b = wdeg.get_weighted_degree("B")
        assert degree_b == 1.0
    
    def test_select_variable(self):
        """Test selección de variable."""
        constraints = [
            Constraint(variables=["A", "B"], predicate=lambda a: True)
        ]
        
        wdeg = WeightedDegreeHeuristic(constraints)
        
        unassigned = ["A", "B", "C"]
        domains = {"A": [1, 2], "B": [1, 2, 3, 4], "C": [1]}
        
        # Debería seleccionar variable con mejor MRV/WDeg ratio
        # C tiene domain_size=1, wdeg=0 (no está en restricciones) -> 1/0.1 = 10
        # A tiene domain_size=2, wdeg=1 -> 2/1 = 2 (mejor)
        selected = wdeg.select_variable(unassigned, domains)
        assert selected in ["A", "C"]  # Ambos son razonables


class TestImpactBasedSearch:
    """Tests para Impact-Based Search."""
    
    def test_initialization(self):
        """Test inicialización."""
        variables = ["A", "B"]
        domains = {"A": [1, 2], "B": [1, 2]}
        constraints = [
            Constraint(
                variables=["A", "B"],
                predicate=lambda a: a.get("A", 0) != a.get("B", 0)
            )
        ]
        
        ibs = ImpactBasedSearch(variables, domains, constraints)
        
        assert len(ibs.impacts) > 0
    
    def test_get_variable_impact(self):
        """Test cálculo de impacto de variable."""
        variables = ["A", "B"]
        domains = {"A": [1, 2], "B": [1, 2]}
        constraints = [
            Constraint(
                variables=["A", "B"],
                predicate=lambda a: a.get("A", 0) != a.get("B", 0)
            )
        ]
        
        ibs = ImpactBasedSearch(variables, domains, constraints)
        
        impact_a = ibs.get_variable_impact("A")
        assert isinstance(impact_a, float)
        assert 0 <= impact_a <= 1


class TestConflictDirectedVariableOrdering:
    """Tests para CDVO."""
    
    def test_initialization(self):
        """Test inicialización."""
        variables = ["A", "B", "C"]
        domains = {"A": [1, 2], "B": [1, 2], "C": [1, 2]}
        constraints = [
            Constraint(variables=["A", "B"], predicate=lambda a: True)
        ]
        
        cdvo = ConflictDirectedVariableOrdering(variables, domains, constraints)
        
        assert cdvo.wdeg is not None
        assert cdvo.ibs is not None
    
    def test_select_variable(self):
        """Test selección de variable."""
        variables = ["A", "B", "C"]
        domains = {"A": [1], "B": [1, 2, 3], "C": [1, 2]}
        constraints = [
            Constraint(variables=["A", "B"], predicate=lambda a: True)
        ]
        
        cdvo = ConflictDirectedVariableOrdering(variables, domains, constraints)
        
        unassigned = ["A", "B", "C"]
        selected = cdvo.select_variable(unassigned, domains)
        
        # Debería seleccionar A (MRV)
        assert selected == "A"


class TestPredicateCache:
    """Tests para Predicate Cache."""
    
    def test_initialization(self):
        """Test inicialización."""
        cache = PredicateCache(max_size=100)
        
        assert cache.max_size == 100
        assert len(cache.cache) == 0
    
    def test_cache_hit(self):
        """Test cache hit."""
        cache = PredicateCache()
        
        def predicate(a):
            return a["x"] > 0
        
        variables = {"x"}
        assignment = {"x": 5}
        
        # Primera llamada: miss
        result1 = cache.get(predicate, assignment, variables)
        assert result1 is None
        
        # Guardar resultado
        cache.put(predicate, assignment, variables, True)
        
        # Segunda llamada: hit
        result2 = cache.get(predicate, assignment, variables)
        assert result2 == True
        assert cache.stats['hits'] == 1
    
    def test_invalidation(self):
        """Test invalidación por variable."""
        cache = PredicateCache()
        
        def predicate(a):
            return a["x"] > 0
        
        variables = {"x"}
        assignment = {"x": 5}
        
        # Guardar resultado
        cache.put(predicate, assignment, variables, True)
        
        # Invalidar variable
        cache.invalidate_variable("x")
        
        # Debería ser miss ahora
        result = cache.get(predicate, assignment, variables)
        assert result is None
    
    def test_eviction(self):
        """Test eviction cuando excede tamaño."""
        cache = PredicateCache(max_size=2)
        
        def pred1(a): return True
        def pred2(a): return True
        def pred3(a): return True
        
        variables = {"x"}
        
        # Llenar caché
        cache.put(pred1, {"x": 1}, variables, True)
        cache.put(pred2, {"x": 2}, variables, True)
        
        assert len(cache.cache) == 2
        
        # Añadir tercero: debería evict el primero
        cache.put(pred3, {"x": 3}, variables, True)
        
        assert len(cache.cache) == 2
        assert cache.stats['evictions'] == 1


class TestCachedPredicate:
    """Tests para Cached Predicate."""
    
    def test_cached_predicate(self):
        """Test predicado cacheado."""
        cache = PredicateCache()
        
        call_count = [0]
        
        def predicate(a):
            call_count[0] += 1
            return a["x"] > 0
        
        variables = {"x"}
        cached_pred = CachedPredicate(predicate, variables, cache)
        
        # Primera llamada
        result1 = cached_pred({"x": 5})
        assert result1 == True
        assert call_count[0] == 1
        
        # Segunda llamada: debería usar caché
        result2 = cached_pred({"x": 5})
        assert result2 == True
        assert call_count[0] == 1  # No incrementó


class TestPredicateCacheManager:
    """Tests para Predicate Cache Manager."""
    
    def test_wrap_predicate(self):
        """Test envolver predicado."""
        manager = PredicateCacheManager()
        
        def predicate(a):
            return a["x"] > 0
        
        variables = {"x"}
        cached = manager.wrap_predicate(predicate, variables)
        
        assert isinstance(cached, CachedPredicate)
    
    def test_notify_assignment(self):
        """Test notificación de asignación."""
        manager = PredicateCacheManager()
        
        def predicate(a):
            return a["x"] > 0
        
        variables = {"x"}
        cached = manager.wrap_predicate(predicate, variables)
        
        # Evaluar y cachear
        result1 = cached({"x": 5})
        
        # Notificar cambio
        manager.notify_assignment("x")
        
        # Caché debería estar invalidado
        stats = manager.get_stats()
        assert stats['invalidations'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""
Tests unitarios para el módulo cubical_types.
"""

import pytest
from lattice_weaver.formal.cubical_types import (
    CubicalFiniteType, CubicalSigmaType, CubicalPredicate, 
    CubicalSubtype, VariableTerm, ValueTerm, CubicalPath
)

class TestCubicalFiniteType:
    def test_creation(self):
        t = CubicalFiniteType(5)
        assert t.size == 5

    def test_to_string(self):
        t = CubicalFiniteType(3)
        assert t.to_string() == "Fin(3)"

    def test_negative_size_raises_error(self):
        with pytest.raises(ValueError):
            CubicalFiniteType(-1)

class TestCubicalSigmaType:
    def test_creation(self):
        t1 = CubicalFiniteType(3)
        t2 = CubicalFiniteType(4)
        sigma = CubicalSigmaType([("x", t1), ("y", t2)])
        assert len(sigma.components) == 2

    def test_to_string(self):
        t1 = CubicalFiniteType(3)
        t2 = CubicalFiniteType(4)
        sigma = CubicalSigmaType([("x", t1), ("y", t2)])
        assert sigma.to_string() == "Σ(x: Fin(3), y: Fin(4))"

class TestCubicalPredicate:
    def test_creation(self):
        left = VariableTerm("x")
        right = ValueTerm(5)
        pred = CubicalPath(left, right)
        assert pred.left == left
        assert pred.right == right

    def test_to_string(self):
        left = VariableTerm("x")
        right = ValueTerm(5)
        pred = CubicalPath(left, right)
        assert pred.to_string() == "Path(x, 5)"

class TestCubicalSubtype:
    def test_creation(self):
        base = CubicalFiniteType(10)
        pred = CubicalPath(VariableTerm("x"), ValueTerm(5))
        subtype = CubicalSubtype(base, pred)
        assert subtype.base_type == base
        assert subtype.predicate == pred

    def test_to_string(self):
        base = CubicalFiniteType(10)
        pred = CubicalPath(VariableTerm("x"), ValueTerm(5))
        subtype = CubicalSubtype(base, pred)
        expected = "{ Fin(10) | Path(x, 5) }"
        assert subtype.to_string() == expected


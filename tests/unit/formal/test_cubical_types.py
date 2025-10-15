"""
Tests unitarios para el módulo cubical_types.
"""

import pytest
from lattice_weaver.formal.cubical_types import (
    CubicalFiniteType, CubicalSigmaType, CubicalPredicate, 
    CubicalSubtype, VariableTerm, ValueTerm, CubicalPath,
    CubicalArithmetic, CubicalComparison
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
        # Note: The order of components is canonicalized, so we test for that
        assert sigma.to_string() == "Σ(x: Fin(3), y: Fin(4))" or sigma.to_string() == "Σ(y: Fin(4), x: Fin(3))"

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

class TestCubicalArithmetic:
    def test_creation(self):
        term1 = VariableTerm("x")
        term2 = ValueTerm(5)
        arith = CubicalArithmetic("sum", (term1, term2))
        assert arith.operation == "sum"
        # The order of terms is canonicalized based on hash
        assert len(arith.terms) == 2

    def test_to_string(self):
        term1 = VariableTerm("x")
        term2 = ValueTerm(5)
        arith = CubicalArithmetic("sum", (term1, term2))
        # The string representation is also canonical (sorted by string)
        assert arith.to_string() == "(5 + x)"

    def test_hash_equality(self):
        term1 = VariableTerm("x")
        term2 = ValueTerm(5)
        arith1 = CubicalArithmetic("sum", (term1, term2))
        arith2 = CubicalArithmetic("sum", (term2, term1))
        assert hash(arith1) == hash(arith2)
        assert arith1 == arith2

class TestCubicalComparison:
    def test_creation(self):
        left_term = VariableTerm("y")
        right_term = ValueTerm(10)
        comp = CubicalComparison(left_term, "==", right_term)
        assert comp.left == left_term
        assert comp.operator == "=="
        assert comp.right == right_term

    def test_to_string(self):
        left_term = VariableTerm("y")
        right_term = ValueTerm(10)
        comp = CubicalComparison(left_term, "==", right_term)
        assert comp.to_string() == "(y == 10)"

    def test_hash_equality(self):
        left_term1 = VariableTerm("y")
        right_term1 = ValueTerm(10)
        comp1 = CubicalComparison(left_term1, "==", right_term1)

        left_term2 = VariableTerm("y")
        right_term2 = ValueTerm(10)
        comp2 = CubicalComparison(left_term2, "==", right_term2)

        assert hash(comp1) == hash(comp2)
        assert comp1 == comp2

        comp3 = CubicalComparison(left_term1, "<=", right_term1)
        assert hash(comp1) != hash(comp3)
        assert comp1 != comp3


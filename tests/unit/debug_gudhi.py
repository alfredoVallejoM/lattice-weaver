import gudhi

def test_gudhi_cycle_betti_numbers():
    st = gudhi.SimplexTree()
    st.insert([0])
    st.insert([1])
    st.insert([2])
    st.insert([0, 1])
    st.insert([1, 2])
    st.insert([0, 2])
    st.compute_persistence()
    betti_numbers = st.betti_numbers()
    print(f"Betti numbers for cycle: {betti_numbers}")
    assert betti_numbers == [1, 1]


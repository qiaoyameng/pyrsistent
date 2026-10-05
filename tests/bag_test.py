import pytest

from pyrsistent import b, pbag, pmap, PBag


class FakeRandom:
    """
    A deterministic random source returning scripted randrange values.
    """
    def __init__(self, values):
        self._values = list(values)
        self.calls = []

    def randrange(self, n):
        self.calls.append(n)
        assert self._values, "FakeRandom exhausted"
        value = self._values.pop(0)
        assert 0 <= value < n
        return value


def test_literalish_works():
    assert b(1, 2) == pbag([1, 2])

def test_empty_bag():
    """
    creating an empty pbag returns a singleton.

    Note that this should NOT be relied upon in application code.
    """
    assert b() is b()

def test_supports_hash():
    assert hash(b(1, 2)) == hash(b(2, 1))

def test_hash_in_dict():
    assert {b(1,2,3,3): "hello"}[b(3,3,2,1)] == "hello"

def test_empty_truthiness():
    assert b(1)
    assert not b()


def test_repr_empty():
    assert repr(b()) == 'pbag([])'

def test_repr_elements():
    assert repr(b(1, 2)) in ('pbag([1, 2])', 'pbag([2, 1])')


def test_add_empty():
    assert b().add(1) == b(1)

def test_remove_final():
    assert b().add(1).remove(1) == b()

def test_remove_nonfinal():
    assert b().add(1).add(1).remove(1) == b(1)

def test_remove_nonexistent():
    with pytest.raises(KeyError) as excinfo:
        b().remove(1)
    assert str(excinfo.exconly()) == 'KeyError: 1'


def test_eq_empty():
    assert b() == b()

def test_neq():
    assert b(1) != b()

def test_eq_same_order():
    assert b(1, 2, 1) == b(1, 2, 1)

def test_eq_different_order():
    assert b(2, 1, 2) == b(1, 2, 2)


def test_count_non_existent():
    assert b().count(1) == 0

def test_count_unique():
    assert b(1).count(1) == 1

def test_count_duplicate():
    assert b(1, 1).count(1) == 2


def test_length_empty():
    assert len(b()) == 0

def test_length_unique():
    assert len(b(1)) == 1

def test_length_duplicates():
    assert len(b(1, 1)) == 2

def test_length_multiple_elements():
    assert len(b(1, 1, 2, 3)) == 4


def test_iter_duplicates():
    assert list(b(1, 1)) == [1, 1]

def test_iter_multiple_elements():
    assert list(b(1, 2, 2)) in ([1, 2, 2], [2, 2, 1])

def test_contains():
    assert 1 in b(1)

def test_not_contains():
    assert 1 not in b(2)

def test_add():
    assert b(3, 3, 3, 2, 2, 1) + b(4, 3, 2, 1) == b(4,
                                                    3, 3, 3, 3,
                                                    2, 2, 2,
                                                    1, 1)

def test_sub():
    assert b(1, 2, 3, 3) - b(3, 4) == b(1, 2, 3)

def test_or():
    assert b(1, 2, 2, 3, 3, 3) | b(1, 2, 3, 4, 4) == b(1,
                                                       2, 2,
                                                       3, 3, 3,
                                                       4, 4)

def test_and():
    assert b(1, 2, 2, 3, 3, 3) & b(2, 3, 3, 4) == b(2, 3, 3)


def test_pbag_is_unorderable():
    with pytest.raises(TypeError):
        _ = b(1) < b(2)  # type: ignore

    with pytest.raises(TypeError):
        _ = b(1) <= b(2)  # type: ignore

    with pytest.raises(TypeError):
        _ = b(1) > b(2)  # type: ignore

    with pytest.raises(TypeError):
        _ = b(1) >= b(2)  # type: ignore


def test_supports_weakref():
    import weakref
    weakref.ref(b(1))


def test_update():
    assert pbag([1, 2, 2]).update([3, 3, 4]) == pbag([1, 2, 2, 3, 3, 4])


def test_update_no_elements():
    b = pbag([1, 2, 2])
    assert b.update([]) is b


def test_iterable():
    """
    PBags can be created from iterables even though they can't be len() hinted.
    """

    assert pbag(iter("a")) == pbag(iter("a"))


def test_sample_with_replacement_interval_boundaries():
    """
    Every randrange value maps to the element owning that interval.
    Counts: a=2, b=1, c=3 -> intervals a:[0,2), b:[2,3), c:[3,6).
    """
    bag = pbag(['a', 'a', 'b', 'c', 'c', 'c'])
    expected = ['a', 'a', 'b', 'c', 'c', 'c']
    for r, element in enumerate(expected):
        rand = FakeRandom([r])
        assert bag.sample_with_replacement(1, rand, key=str) == [element]
        assert rand.calls == [6]


def test_sample_with_replacement_honors_key_order():
    """
    The key fixes the interval order: here c < b < a, so the intervals
    become c:[0,3), b:[3,4), a:[4,6).
    """
    bag = pbag(['a', 'a', 'b', 'c', 'c', 'c'])
    key = lambda e: {'c': 0, 'b': 1, 'a': 2}[e]
    expected = ['c', 'c', 'c', 'b', 'a', 'a']
    for r, element in enumerate(expected):
        rand = FakeRandom([r])
        assert bag.sample_with_replacement(1, rand, key=key) == [element]


def test_sample_with_replacement_key_ties_broken_by_repr():
    """
    Elements with equal keys are ordered by repr, so 1 sorts before 2
    and gets interval [0,2) while 2 gets [2,3).
    """
    bag = pbag([1, 1, 2])
    rand = FakeRandom([2])
    assert bag.sample_with_replacement(1, rand, key=lambda e: 0) == [2]
    rand = FakeRandom([1])
    assert bag.sample_with_replacement(1, rand, key=lambda e: 0) == [1]


def test_sample_with_replacement_default_order_is_repr():
    bag = pbag([10, 2])
    # repr order: '10' < '2', so 10 owns [0, 1) and 2 owns [1, 2)
    assert bag.sample_with_replacement(1, FakeRandom([0])) == [10]
    assert bag.sample_with_replacement(1, FakeRandom([1])) == [2]


def test_sample_with_replacement_exhaustive_weights():
    """
    Exhaustively drawing with every possible randrange value reproduces
    the weights exactly, including independence across draws.
    """
    bag = pbag(['x', 'y', 'y'])
    single = {}
    for r in range(3):
        (element,) = bag.sample_with_replacement(1, FakeRandom([r]))
        single[element] = single.get(element, 0) + 1
    assert single == {'x': 1, 'y': 2}

    pairs = {}
    for r1 in range(3):
        for r2 in range(3):
            drawn = tuple(bag.sample_with_replacement(2, FakeRandom([r1, r2])))
            pairs[drawn] = pairs.get(drawn, 0) + 1
    assert pairs == {('x', 'x'): 1, ('x', 'y'): 2,
                     ('y', 'x'): 2, ('y', 'y'): 4}


def test_sample_with_replacement_k_zero():
    rand = FakeRandom([])
    assert pbag([1, 1, 2]).sample_with_replacement(0, rand) == []
    assert rand.calls == []


def test_sample_with_replacement_negative_k():
    with pytest.raises(ValueError):
        pbag([1]).sample_with_replacement(-1, FakeRandom([]))


def test_sample_with_replacement_empty_bag():
    with pytest.raises(ValueError):
        pbag([]).sample_with_replacement(1, FakeRandom([]))


def test_sample_without_replacement_returns_selection_and_new_bag():
    bag = pbag(['a', 'a', 'b', 'c', 'c', 'c'])
    rand = FakeRandom([2, 3])
    selected, rest = bag.sample_without_replacement(2, rand, key=str)
    assert selected == ['b', 'c']
    assert rest == pbag(['a', 'a', 'c', 'c'])
    assert rand.calls == [6, 5]
    # The original bag is unchanged.
    assert bag == pbag(['a', 'a', 'b', 'c', 'c', 'c'])


def test_sample_without_replacement_exhaustive():
    """
    Exhaustively drawing two elements from pbag([1, 1, 2]) without
    replacement yields results consistent with the remaining bag and
    with the count-proportional weights.
    """
    bag = pbag([1, 1, 2])
    outcomes = {}
    for r1 in range(3):
        for r2 in range(2):
            selected, rest = bag.sample_without_replacement(
                2, FakeRandom([r1, r2]))
            assert bag - pbag(selected) == rest
            assert len(rest) == 1
            outcomes[tuple(selected)] = outcomes.get(tuple(selected), 0) + 1
    # First draw: 1 with probability 2/3, 2 with probability 1/3.
    # Second draw is uniform over what remains.
    assert outcomes == {(1, 1): 2, (1, 2): 2, (2, 1): 2}


def test_sample_without_replacement_all_elements():
    bag = pbag([1, 1, 2])
    selected, rest = bag.sample_without_replacement(
        3, FakeRandom([0, 0, 0]))
    assert selected == [1, 1, 2]
    assert rest == pbag([])
    assert bag == pbag([1, 1, 2])


def test_sample_without_replacement_too_many_rejected_upfront():
    """
    Requesting more elements than the bag holds fails before the random
    source is consulted.
    """
    rand = FakeRandom([])
    with pytest.raises(ValueError):
        pbag([1, 2]).sample_without_replacement(3, rand)
    assert rand.calls == []


def test_sample_without_replacement_k_zero():
    bag = pbag([1, 2])
    rand = FakeRandom([])
    selected, rest = bag.sample_without_replacement(0, rand)
    assert selected == []
    assert rest is bag
    assert rand.calls == []


def test_sample_without_replacement_negative_k():
    with pytest.raises(ValueError):
        pbag([1]).sample_without_replacement(-1, FakeRandom([]))


def _huge_bag():
    # Counts above 2**53 cannot be represented exactly as floats.
    return PBag(pmap({'a': 2**53 + 1, 'b': 2**60}))


def test_sample_with_replacement_huge_counts_exact():
    """
    Integer weights keep counts above 2**53 exact: 'a' owns
    [0, 2**53 + 1) and 'b' owns [2**53 + 1, 2**53 + 1 + 2**60).
    """
    bag = _huge_bag()
    total = 2**53 + 1 + 2**60
    assert len(bag) == total
    # Boundary values around the end of a's interval. A float based
    # implementation could not distinguish these.
    assert bag.sample_with_replacement(1, FakeRandom([2**53])) == ['a']
    assert bag.sample_with_replacement(1, FakeRandom([2**53 + 1])) == ['b']
    assert bag.sample_with_replacement(1, FakeRandom([total - 1])) == ['b']


def test_sample_without_replacement_huge_counts_exact():
    bag = _huge_bag()
    selected, rest = bag.sample_without_replacement(
        1, FakeRandom([2**53 + 1]))
    assert selected == ['b']
    assert rest.count('a') == 2**53 + 1
    assert rest.count('b') == 2**60 - 1
    # The original bag is unchanged.
    assert bag.count('a') == 2**53 + 1
    assert bag.count('b') == 2**60

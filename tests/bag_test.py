import pytest

from pyrsistent import b, pbag
from pyrsistent import pmap
from pyrsistent._pbag import PBag


class FakeRandom:
    """A scripted random source recording every randrange request."""

    def __init__(self, rolls):
        self._rolls = list(rolls)
        self.requests = []

    def randrange(self, n):
        self.requests.append(n)
        roll = self._rolls.pop(0)
        assert 0 <= roll < n
        return roll


def weighted_bag():
    # Intervals with key=str: 'a' -> [0, 2), 'b' -> [2, 5), 'c' -> [5, 10)
    return pbag(['a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c'])


def test_sample_with_replacement_interval_boundaries():
    expected = ['a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c']
    rand = FakeRandom(range(10))
    assert weighted_bag().sample_with_replacement(10, rand, key=str) == expected
    assert rand.requests == [10] * 10


def test_sample_with_replacement_exhaustive_weights():
    # Every possible single draw roll reproduces the exact weights.
    tally = {'a': 0, 'b': 0, 'c': 0}
    for roll in range(10):
        picked = weighted_bag().sample_with_replacement(1, FakeRandom([roll]), key=str)[0]
        tally[picked] += 1
    assert tally == {'a': 2, 'b': 3, 'c': 5}


def test_sample_with_replacement_does_not_modify_bag():
    bag = weighted_bag()
    bag.sample_with_replacement(5, FakeRandom([0] * 5), key=str)
    assert bag == weighted_bag()


def test_sample_with_replacement_zero_draws():
    assert weighted_bag().sample_with_replacement(0, FakeRandom([]), key=str) == []
    assert b().sample_with_replacement(0, FakeRandom([])) == []


def test_sample_with_replacement_empty_bag():
    with pytest.raises(ValueError):
        b().sample_with_replacement(1, FakeRandom([]))


def test_sample_with_replacement_negative_k():
    with pytest.raises(ValueError):
        weighted_bag().sample_with_replacement(-1, FakeRandom([]))


def test_sample_with_replacement_key_orders_intervals():
    bag = pbag(['bb', 'a'])
    assert bag.sample_with_replacement(2, FakeRandom([0, 1]), key=len) == ['a', 'bb']
    assert bag.sample_with_replacement(2, FakeRandom([0, 1]), key=lambda e: -len(e)) == ['bb', 'a']


def test_sample_key_ties_keep_internal_order():
    elements = ['x', 1, (2,), 'y', 3]
    bag = pbag(elements)
    internal_order = [e for e, _ in bag._counts.iteritems()]
    rand = FakeRandom(range(len(elements)))
    assert bag.sample_with_replacement(len(elements), rand, key=lambda e: 0) == internal_order


def test_sample_without_replacement_returns_new_bag():
    bag = weighted_bag()
    selected, new_bag = bag.sample_without_replacement(4, FakeRandom([0, 0, 0, 0]), key=str)
    assert selected == ['a', 'a', 'b', 'b']
    assert new_bag == pbag(['b', 'c', 'c', 'c', 'c', 'c'])
    assert bag == weighted_bag()


def test_sample_without_replacement_interval_boundaries():
    # First draw on 'a':2, 'b':3 with total 5, then on the remainder.
    bag = pbag(['a', 'a', 'b', 'b', 'b'])
    assert bag.sample_without_replacement(2, FakeRandom([0, 0]), key=str)[0] == ['a', 'a']
    assert bag.sample_without_replacement(2, FakeRandom([1, 0]), key=str)[0] == ['a', 'a']
    assert bag.sample_without_replacement(2, FakeRandom([1, 1]), key=str)[0] == ['a', 'b']
    assert bag.sample_without_replacement(2, FakeRandom([2, 0]), key=str)[0] == ['b', 'a']
    assert bag.sample_without_replacement(2, FakeRandom([4, 3]), key=str)[0] == ['b', 'b']


def test_sample_without_replacement_exhaustive_weights():
    # All roll combinations for two draws from 'a':1, 'b':2.
    bag = pbag(['a', 'b', 'b'])
    outcomes = {}
    for first in range(3):
        for second in range(2):
            selected, new_bag = bag.sample_without_replacement(
                2, FakeRandom([first, second]), key=str)
            assert len(selected) == 2
            assert len(new_bag) == 1
            remaining = next(iter(new_bag))
            assert sorted(selected + [remaining]) == ['a', 'b', 'b']
            outcomes[tuple(selected)] = outcomes.get(tuple(selected), 0) + 1
    # 6 equally sized roll cells: a then b (2 cells), b then a (2), b then b (2)
    assert outcomes == {('a', 'b'): 2, ('b', 'a'): 2, ('b', 'b'): 2}


def test_sample_without_replacement_draw_all_empties_bag():
    bag = weighted_bag()
    selected, new_bag = bag.sample_without_replacement(10, FakeRandom([0] * 10), key=str)
    assert sorted(selected) == sorted(list(bag))
    assert new_bag == b()


def test_sample_without_replacement_zero_draws():
    bag = weighted_bag()
    selected, new_bag = bag.sample_without_replacement(0, FakeRandom([]))
    assert selected == []
    assert new_bag is bag


def test_sample_without_replacement_rejects_oversized_draw_before_starting():
    rand = FakeRandom([])
    with pytest.raises(ValueError):
        weighted_bag().sample_without_replacement(11, rand, key=str)
    assert rand.requests == []


def test_sample_without_replacement_negative_k():
    with pytest.raises(ValueError):
        weighted_bag().sample_without_replacement(-1, FakeRandom([]))


def big_count_bag():
    big = 2 ** 60
    # Intervals with key=str: 'a' -> [0, big), 'b' -> [big, 2*big + 1),
    # 'c' -> [2*big + 1, 2*big + 8)
    return PBag(pmap({'a': big, 'b': big + 1, 'c': 7})), big


def test_sample_with_replacement_huge_counts_exact_boundaries():
    bag, big = big_count_bag()
    total = 2 * big + 8
    rolls = [0, big - 1, big, 2 * big, 2 * big + 1, total - 1]
    expected = ['a', 'a', 'b', 'b', 'c', 'c']
    assert bag.sample_with_replacement(6, FakeRandom(rolls), key=str) == expected


def test_sample_without_replacement_huge_counts_stay_exact():
    bag, big = big_count_bag()
    selected, new_bag = bag.sample_without_replacement(2, FakeRandom([big, 0]), key=str)
    assert selected == ['b', 'a']
    assert new_bag.count('a') == big - 1
    assert new_bag.count('b') == big
    assert new_bag.count('c') == 7
    assert bag.count('a') == big


def test_sample_huge_counts_cost_depends_on_distinct_elements_only():
    # Drawing from counts around 2**70 must not materialize count-sized
    # structures; only one randrange call per draw is made.
    bag = PBag(pmap({'only': 2 ** 70 + 123}))
    rand = FakeRandom([0] * 100)
    assert bag.sample_with_replacement(100, rand, key=str) == ['only'] * 100
    assert rand.requests == [2 ** 70 + 123] * 100


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

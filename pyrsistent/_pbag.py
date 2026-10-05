from bisect import bisect_right
from collections.abc import Container, Iterable, Sized, Hashable
from functools import reduce
from typing import Generic, TypeVar
from pyrsistent._pmap import pmap

T_co = TypeVar('T_co', covariant=True)


def _add_to_counters(counters, element):
    return counters.set(element, counters.get(element, 0) + 1)


def _ordered_elements(counts, key):
    """
    Return the distinct elements of counts in a deterministic order.

    Elements are ordered by ``key(element)`` if a key is given, otherwise by
    ``repr(element)``. Ties are broken by ``repr(element)`` so that the order
    is stable even when the key maps different elements to the same value.
    """
    if key is None:
        return sorted(counts, key=repr)
    return sorted(counts, key=lambda elt: (key(elt), repr(elt)))


def _cumulative_weights(elements, counts):
    """
    Return (elements_with_positive_count, cumulative_endpoints, total).
    """
    weighted = []
    endpoints = []
    total = 0
    for elt in elements:
        count = counts.get(elt, 0)
        if count > 0:
            total += count
            weighted.append(elt)
            endpoints.append(total)
    return weighted, endpoints, total


def _pick_weighted(weighted, endpoints, total, rand):
    """
    Pick one element proportional to its weight using rand.randrange(total).

    All arithmetic is done on integers so weights of any magnitude (including
    counts above 2**53) are handled exactly.
    """
    r = rand.randrange(total)
    return weighted[bisect_right(endpoints, r)]


class PBag(Generic[T_co]):
    """
    A persistent bag/multiset type.

    Requires elements to be hashable, and allows duplicates, but has no
    ordering. Bags are hashable.

    Do not instantiate directly, instead use the factory functions :py:func:`b`
    or :py:func:`pbag` to create an instance.

    Some examples:

    >>> s = pbag([1, 2, 3, 1])
    >>> s2 = s.add(4)
    >>> s3 = s2.remove(1)
    >>> s
    pbag([1, 1, 2, 3])
    >>> s2
    pbag([1, 1, 2, 3, 4])
    >>> s3
    pbag([1, 2, 3, 4])
    """

    __slots__ = ('_counts', '__weakref__')

    def __init__(self, counts):
        self._counts = counts

    def add(self, element):
        """
        Add an element to the bag.

        >>> s = pbag([1])
        >>> s2 = s.add(1)
        >>> s3 = s.add(2)
        >>> s2
        pbag([1, 1])
        >>> s3
        pbag([1, 2])
        """
        return PBag(_add_to_counters(self._counts, element))

    def update(self, iterable):
        """
        Update bag with all elements in iterable.

        >>> s = pbag([1])
        >>> s.update([1, 2])
        pbag([1, 1, 2])
        """
        if iterable:
            return PBag(reduce(_add_to_counters, iterable, self._counts))

        return self

    def remove(self, element):
        """
        Remove an element from the bag.

        >>> s = pbag([1, 1, 2])
        >>> s2 = s.remove(1)
        >>> s3 = s.remove(2)
        >>> s2
        pbag([1, 2])
        >>> s3
        pbag([1, 1])
        """
        if element not in self._counts:
            raise KeyError(element)
        elif self._counts[element] == 1:
            newc = self._counts.remove(element)
        else:
            newc = self._counts.set(element, self._counts[element] - 1)
        return PBag(newc)

    def count(self, element):
        """
        Return the number of times an element appears.


        >>> pbag([]).count('non-existent')
        0
        >>> pbag([1, 1, 2]).count(1)
        2
        """
        return self._counts.get(element, 0)

    def sample_with_replacement(self, k, rand, key=None):
        """
        Return a list of ``k`` elements sampled from this bag with
        replacement, where each element is selected with probability
        proportional to its count. The bag is not modified and no
        intermediate structure proportional to the total count is created,
        so bags with very large counts are sampled in O(distinct elements).

        ``rand`` is the random source, supplied by the caller. It may be any
        object providing ``randrange(n)``, e.g. an instance of
        ``random.Random``. Sampling uses integer weights only, so counts
        above 2**53 are handled exactly.

        ``key`` is an optional function mapping each element to a sort key.
        It fixes the order in which elements are assigned their sampling
        intervals, making the result reproducible for a given random source.
        Ties are broken by ``repr(element)``. If omitted, elements are
        ordered by ``repr(element)``.

        >>> import random
        >>> s = pbag([1, 1, 2])
        >>> s.sample_with_replacement(3, random.Random(42))
        [2, 1, 1]
        """
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0:
            return []
        elements = _ordered_elements(self._counts, key)
        weighted, endpoints, total = _cumulative_weights(elements, self._counts)
        if total == 0:
            raise ValueError("Cannot sample from an empty bag")
        return [_pick_weighted(weighted, endpoints, total, rand)
                for _ in range(k)]

    def sample_without_replacement(self, k, rand, key=None):
        """
        Sample ``k`` elements from this bag without replacement and return a
        tuple ``(selected, new_bag)`` where ``selected`` is a list of the
        sampled elements and ``new_bag`` is a new PBag with those elements
        removed. This bag is left unchanged. Each remaining element is
        selected with probability proportional to its current count.

        If ``k`` exceeds the number of elements in the bag a ValueError is
        raised before any sampling (and before the random source is
        consulted). Sampling uses integer weights only, so counts above
        2**53 are handled exactly, and no intermediate structure
        proportional to the total count is created.

        ``rand`` and ``key`` behave as in :py:meth:`sample_with_replacement`.

        >>> import random
        >>> s = pbag([1, 1, 2])
        >>> selected, rest = s.sample_without_replacement(2, random.Random(42))
        >>> selected
        [2, 1]
        >>> rest
        pbag([1])
        >>> s
        pbag([1, 1, 2])
        """
        if k < 0:
            raise ValueError("k must be non-negative")
        total = len(self)
        if k > total:
            raise ValueError(
                "Cannot sample {0} elements without replacement from a bag "
                "of size {1}".format(k, total))
        if k == 0:
            return [], self
        elements = _ordered_elements(self._counts, key)
        remaining = {elt: count for elt, count in self._counts.iteritems()}
        selected = []
        for _ in range(k):
            weighted, endpoints, remaining_total = _cumulative_weights(
                elements, remaining)
            elt = _pick_weighted(weighted, endpoints, remaining_total, rand)
            selected.append(elt)
            remaining[elt] -= 1
            if remaining[elt] == 0:
                del remaining[elt]
        return selected, PBag(pmap(remaining))

    def __len__(self):
        """
        Return the length including duplicates.

        >>> len(pbag([1, 1, 2]))
        3
        """
        return sum(self._counts.itervalues())

    def __iter__(self):
        """
        Return an iterator of all elements, including duplicates.

        >>> list(pbag([1, 1, 2]))
        [1, 1, 2]
        >>> list(pbag([1, 2]))
        [1, 2]
        """
        for elt, count in self._counts.iteritems():
            for i in range(count):
                yield elt

    def __contains__(self, elt):
        """
        Check if an element is in the bag.

        >>> 1 in pbag([1, 1, 2])
        True
        >>> 0 in pbag([1, 2])
        False
        """
        return elt in self._counts

    def __repr__(self):
        return "pbag({0})".format(list(self))

    def __eq__(self, other):
        """
        Check if two bags are equivalent, honoring the number of duplicates,
        and ignoring insertion order.

        >>> pbag([1, 1, 2]) == pbag([1, 2])
        False
        >>> pbag([2, 1, 0]) == pbag([0, 1, 2])
        True
        """
        if type(other) is not PBag:
            raise TypeError("Can only compare PBag with PBags")
        return self._counts == other._counts

    def __lt__(self, other):
        raise TypeError('PBags are not orderable')

    __le__ = __lt__
    __gt__ = __lt__
    __ge__ = __lt__

    # Multiset-style operations similar to collections.Counter

    def __add__(self, other):
        """
        Combine elements from two PBags.

        >>> pbag([1, 2, 2]) + pbag([2, 3, 3])
        pbag([1, 2, 2, 2, 3, 3])
        """
        if not isinstance(other, PBag):
            return NotImplemented
        result = self._counts.evolver()
        for elem, other_count in other._counts.iteritems():
            result[elem] = self.count(elem) + other_count
        return PBag(result.persistent())

    def __sub__(self, other):
        """
        Remove elements from one PBag that are present in another.

        >>> pbag([1, 2, 2, 2, 3]) - pbag([2, 3, 3, 4])
        pbag([1, 2, 2])
        """
        if not isinstance(other, PBag):
            return NotImplemented
        result = self._counts.evolver()
        for elem, other_count in other._counts.iteritems():
            newcount = self.count(elem) - other_count
            if newcount > 0:
                result[elem] = newcount
            elif elem in self:
                result.remove(elem)
        return PBag(result.persistent())

    def __or__(self, other):
        """
        Union: Keep elements that are present in either of two PBags.

        >>> pbag([1, 2, 2, 2]) | pbag([2, 3, 3])
        pbag([1, 2, 2, 2, 3, 3])
        """
        if not isinstance(other, PBag):
            return NotImplemented
        result = self._counts.evolver()
        for elem, other_count in other._counts.iteritems():
            count = self.count(elem)
            newcount = max(count, other_count)
            result[elem] = newcount
        return PBag(result.persistent())

    def __and__(self, other):
        """
        Intersection: Only keep elements that are present in both PBags.

        >>> pbag([1, 2, 2, 2]) & pbag([2, 3, 3])
        pbag([2])
        """
        if not isinstance(other, PBag):
            return NotImplemented
        result = pmap().evolver()
        for elem, count in self._counts.iteritems():
            newcount = min(count, other.count(elem))
            if newcount > 0:
                result[elem] = newcount
        return PBag(result.persistent())

    def __hash__(self):
        """
        Hash based on value of elements.

        >>> m = pmap({pbag([1, 2]): "it's here!"})
        >>> m[pbag([2, 1])]
        "it's here!"
        >>> pbag([1, 1, 2]) in m
        False
        """
        return hash(self._counts)


Container.register(PBag)
Iterable.register(PBag)
Sized.register(PBag)
Hashable.register(PBag)


def b(*elements):
    """
    Construct a persistent bag.

    Takes an arbitrary number of arguments to insert into the new persistent
    bag.

    >>> b(1, 2, 3, 2)
    pbag([1, 2, 2, 3])
    """
    return pbag(elements)


def pbag(elements):
    """
    Convert an iterable to a persistent bag.

    Takes an iterable with elements to insert.

    >>> pbag([1, 2, 3, 2])
    pbag([1, 2, 2, 3])
    """
    if not elements:
        return _EMPTY_PBAG
    return PBag(reduce(_add_to_counters, elements, pmap()))


_EMPTY_PBAG = PBag(pmap())

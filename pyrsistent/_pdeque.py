from collections.abc import Sequence, Hashable
from itertools import islice, chain
from numbers import Integral
from typing import TypeVar, Generic

T_co = TypeVar('T_co', covariant=True)


class _StepCounter(object):
    """
    Test instrumentation counting the basic O(1) steps (stream cell creations
    and lazy cell forcings) performed inside the deque implementation.

    Disabled by default and cheap to keep around: tests enable it to assert
    that every single operation performs only a bounded amount of work,
    instead of relying on average timings to rule out long tails.
    """
    __slots__ = ('count', 'enabled')

    def __init__(self):
        self.count = 0
        self.enabled = False

    def reset(self):
        self.count = 0

    def step(self, n=1):
        if self.enabled:
            self.count += n


#: Shared counter used by the tests to measure per operation work.
step_counter = _StepCounter()


class _Stream(object):
    """
    Lazy, memoized stream used as the basic building block of the deque.

    The node of a stream is computed on first access and the result is cached
    in the stream cell itself. Since cells are shared by all deque versions
    derived from the same ancestors, any delayed (incremental) rebalancing
    work is performed at most once, no matter how many branches end up forcing
    it. Streams are immutable from the outside; the caching is an internal,
    semantically invisible optimization.
    """
    __slots__ = ('_thunk', '_node')

    def __init__(self, thunk, node):
        self._thunk = thunk
        self._node = node

    @staticmethod
    def cons(head, tail):
        step_counter.step()
        return _Stream(None, (head, tail))

    @staticmethod
    def delay(thunk):
        step_counter.step()
        return _Stream(thunk, None)

    def _force(self):
        node = self._node
        if node is None:
            step_counter.step()
            node = self._thunk()
            self._thunk = None
            if node is None:
                node = ()
            self._node = node
        return node

    def is_empty(self):
        return not self._force()

    @property
    def head(self):
        return self._force()[0]

    @property
    def tail(self):
        return self._force()[1]


_EMPTY_STREAM = _Stream(None, ())


def _take(count, stream):
    """
    Lazy stream of the first count elements of stream. Realizing each node
    requires only a constant amount of new work.
    """
    if count <= 0:
        return _EMPTY_STREAM

    def thunk():
        if stream.is_empty():
            return None
        return stream.head, _take(count - 1, stream.tail)

    return _Stream.delay(thunk)


def _rot(y, x, skip):
    """
    Lazy stream containing the elements of y followed by the elements of x
    beyond the first skip ones, in reverse order:

        y ++ reverse(drop(skip, x))

    The reversal is performed incrementally: realizing each node performs at
    most two basic steps on x, so no single forcing ever scans more than a
    constant number of elements. The caller guarantees
    len(x) <= 2 * len(y) + 3, which keeps the eager finish (once y runs out)
    constantly bounded.
    """
    return _Stream.delay(lambda: _rot_node(y, x, skip, _EMPTY_STREAM))


def _rot_node(y, x, skip, acc):
    if not y.is_empty():
        for _ in range(2):
            if not x.is_empty():
                if skip > 0:
                    x = x.tail
                    skip -= 1
                else:
                    acc = _Stream.cons(x.head, acc)
                    x = x.tail
        return y.head, _Stream.delay(lambda: _rot_node(y.tail, x, skip, acc))

    # y is exhausted, finish the (constantly bounded) remainder of x eagerly.
    while skip > 0 and not x.is_empty():
        x = x.tail
        skip -= 1
    while not x.is_empty():
        acc = _Stream.cons(x.head, acc)
        x = x.tail

    node = acc._force()
    return node if node else None


def _exec2(stream):
    """
    Force the next two nodes of a schedule stream and return the remainder.

    Every endpoint operation runs this on the pending schedules to guarantee
    that delayed rebalancing work is fully realized before it can be needed,
    which is what makes the per operation work bounded in the worst case and
    not only on average.
    """
    if not stream.is_empty():
        stream = stream.tail
        if not stream.is_empty():
            stream = stream.tail
    return stream


def _rebalance(left, left_sched, left_len, right, right_sched, right_len):
    """
    Restore the invariant that neither side may be more than twice as long as
    the other (plus one). The actual rotation work is delayed inside lazy
    streams; the fresh streams themselves serve as schedules for that work.
    """
    if left_len > 2 * right_len + 1:
        n = left_len + right_len
        i = n // 2
        new_left = _take(i, left)
        new_right = _rot(right, left, i)
        return new_left, new_left, i, new_right, new_right, n - i

    if right_len > 2 * left_len + 1:
        n = left_len + right_len
        j = n // 2
        new_right = _take(j, right)
        new_left = _rot(left, right, j)
        return new_left, new_left, n - j, new_right, new_right, j

    return left, left_sched, left_len, right, right_sched, right_len


def _stream_iter(stream):
    while not stream.is_empty():
        yield stream.head
        stream = stream.tail


def _realized(stream):
    """
    Force all pending nodes of stream and return it. Linear in the length of
    the stream; the realized nodes are shared with all versions holding them.
    """
    current = stream
    while not current.is_empty():
        current = current.tail
    return stream


def _stream_reverse(stream):
    result = _EMPTY_STREAM
    while not stream.is_empty():
        result = _Stream.cons(stream.head, result)
        stream = stream.tail
    return result


def _stream_remove_first(stream, elem):
    prefix = []
    current = stream
    while not current.is_empty():
        if current.head == elem:
            result = current.tail
            for value in reversed(prefix):
                result = _Stream.cons(value, result)
            return result
        prefix.append(current.head)
        current = current.tail
    raise ValueError('{0} not found in stream'.format(elem))


def _push_left(state, elem):
    left, left_sched, left_len, right, right_sched, right_len = state
    return _Stream.cons(elem, left), left_sched, left_len + 1, right, right_sched, right_len


def _push_right(state, elem):
    left, left_sched, left_len, right, right_sched, right_len = state
    return left, left_sched, left_len, _Stream.cons(elem, right), right_sched, right_len + 1


def _pop_left(state):
    left, left_sched, left_len, right, right_sched, right_len = state
    if left_len > 0:
        return left.tail, left_sched, left_len - 1, right, right_sched, right_len
    if right_len > 0:
        # The balance invariant guarantees that the right side holds at most
        # one element when the left side is empty.
        return left, left_sched, left_len, right.tail, right_sched, right_len - 1
    return state


def _pop_right(state):
    left, left_sched, left_len, right, right_sched, right_len = state
    if right_len > 0:
        return left, left_sched, left_len, right.tail, right_sched, right_len - 1
    if left_len > 0:
        # See _pop_left.
        return left.tail, left_sched, left_len - 1, right, right_sched, right_len
    return state


class PDeque(Generic[T_co]):
    """
    Persistent double ended queue (deque). Allows quick appends and pops in both ends.

    Implemented using two lazy, memoized streams with smooth (incremental)
    rebalancing: whenever one side grows too long relative to the other, the
    rebalancing rotation is spread out over the following operations instead of
    being performed in one burst. Every endpoint operation therefore performs
    only a bounded amount of work. Pending work is cached in shared stream
    cells, so it is safely shared between all versions (branches) derived from
    the same deque; no branch ever re-does a reversal another branch already
    paid for.

    A maximum length can be specified to create a bounded queue.

    Fully supports the Sequence and Hashable protocols including indexing and slicing but
    if you need fast random access go for the PVector instead.

    Do not instantiate directly, instead use the factory functions :py:func:`dq` or :py:func:`pdeque` to
    create an instance.

    Some examples:

    >>> x = pdeque([1, 2, 3])
    >>> x.left
    1
    >>> x.right
    3
    >>> x[0] == x.left
    True
    >>> x[-1] == x.right
    True
    >>> x.pop()
    pdeque([1, 2])
    >>> x.pop() == x[:-1]
    True
    >>> x.popleft()
    pdeque([2, 3])
    >>> x.append(4)
    pdeque([1, 2, 3, 4])
    >>> x.appendleft(4)
    pdeque([4, 1, 2, 3])

    >>> y = pdeque([1, 2, 3], maxlen=3)
    >>> y.append(4)
    pdeque([2, 3, 4], maxlen=3)
    >>> y.appendleft(4)
    pdeque([4, 1, 2], maxlen=3)
    """
    __slots__ = ('_left_stream', '_left_sched', '_left_length',
                 '_right_stream', '_right_sched', '_right_length',
                 '_maxlen', '__weakref__')

    def __new__(cls, left_stream, right_stream, left_length, right_length, maxlen=None,
                left_sched=None, right_sched=None):
        instance = super(PDeque, cls).__new__(cls)
        instance._left_stream = left_stream
        instance._right_stream = right_stream
        instance._left_length = left_length
        instance._right_length = right_length
        instance._left_sched = _EMPTY_STREAM if left_sched is None else left_sched
        instance._right_sched = _EMPTY_STREAM if right_sched is None else right_sched

        if maxlen is not None:
            if not isinstance(maxlen, Integral):
                raise TypeError('An integer is required as maxlen')

            if maxlen < 0:
                raise ValueError("maxlen must be non-negative")

        instance._maxlen = maxlen
        return instance

    def _state(self):
        return (self._left_stream, self._left_sched, self._left_length,
                self._right_stream, self._right_sched, self._right_length)

    @staticmethod
    def _from_state(state, maxlen):
        left, left_sched, left_len, right, right_sched, right_len = _rebalance(*state)
        return PDeque(left, right, left_len, right_len, maxlen,
                      left_sched=_exec2(left_sched), right_sched=_exec2(right_sched))

    @property
    def right(self):
        """
        Rightmost element in dqueue.
        """
        if self._right_length:
            return self._right_stream.head

        if self._left_length:
            # The balance invariant guarantees a single element on this side.
            return self._left_stream.head

        raise IndexError('No elements in empty deque')

    @property
    def left(self):
        """
        Leftmost element in dqueue.
        """
        if self._left_length:
            return self._left_stream.head

        if self._right_length:
            # The balance invariant guarantees a single element on this side.
            return self._right_stream.head

        raise IndexError('No elements in empty deque')

    def __iter__(self):
        return chain(_stream_iter(self._left_stream),
                     reversed(list(_stream_iter(self._right_stream))))

    def __repr__(self):
        return "pdeque({0}{1})".format(list(self),
                                       ', maxlen={0}'.format(self._maxlen) if self._maxlen is not None else '')
    __str__ = __repr__

    @property
    def maxlen(self):
        """
        Maximum length of the queue.
        """
        return self._maxlen

    def pop(self, count=1):
        """
        Return new deque with rightmost element removed. Popping the empty queue
        will return the empty queue. A optional count can be given to indicate the
        number of elements to pop. Popping with a negative index is the same as
        popleft. Executes in O(k) where k is the number of elements to pop, with
        only a bounded amount of work performed by each single element pop.

        >>> pdeque([1, 2]).pop()
        pdeque([1])
        >>> pdeque([1, 2]).pop(2)
        pdeque([])
        >>> pdeque([1, 2]).pop(-1)
        pdeque([2])
        """
        if count < 0:
            return self.popleft(-count)

        result = self
        while count > 0 and len(result) > 0:
            result = result._from_state(_pop_right(result._state()), result._maxlen)
            count -= 1

        return result

    def popleft(self, count=1):
        """
        Return new deque with leftmost element removed. Otherwise functionally
        equivalent to pop().

        >>> pdeque([1, 2]).popleft()
        pdeque([2])
        """
        if count < 0:
            return self.pop(-count)

        result = self
        while count > 0 and len(result) > 0:
            result = result._from_state(_pop_left(result._state()), result._maxlen)
            count -= 1

        return result

    def _is_empty(self):
        return self._left_length == 0 and self._right_length == 0

    def __lt__(self, other):
        if not isinstance(other, PDeque):
            return NotImplemented

        return tuple(self) < tuple(other)

    def __eq__(self, other):
        if not isinstance(other, PDeque):
            return NotImplemented

        if tuple(self) == tuple(other):
            # Sanity check of the length value since it is redundant (there for performance)
            assert len(self) == len(other)
            return True

        return False

    def __hash__(self):
        return hash(tuple(self))

    def __len__(self):
        return self._left_length + self._right_length

    def append(self, elem):
        """
        Return new deque with elem as the rightmost element.

        >>> pdeque([1, 2]).append(3)
        pdeque([1, 2, 3])
        """
        if self._maxlen is not None and len(self) == self._maxlen:
            if self._maxlen == 0:
                return self
            return self.popleft().append(elem)

        return self._from_state(_push_right(self._state(), elem), self._maxlen)

    def appendleft(self, elem):
        """
        Return new deque with elem as the leftmost element.

        >>> pdeque([1, 2]).appendleft(3)
        pdeque([3, 1, 2])
        """
        if self._maxlen is not None and len(self) == self._maxlen:
            if self._maxlen == 0:
                return self
            return self.pop().appendleft(elem)

        return self._from_state(_push_left(self._state(), elem), self._maxlen)

    def extend(self, iterable):
        """
        Return new deque with all elements of iterable appended to the right.

        >>> pdeque([1, 2]).extend([3, 4])
        pdeque([1, 2, 3, 4])
        """
        result = self
        for elem in iterable:
            result = result.append(elem)

        return result

    def extendleft(self, iterable):
        """
        Return new deque with all elements of iterable appended to the left.

        NB! The elements will be inserted in reverse order compared to the order in the iterable.

        >>> pdeque([1, 2]).extendleft([3, 4])
        pdeque([4, 3, 1, 2])
        """
        result = self
        for elem in iterable:
            result = result.appendleft(elem)

        return result

    def count(self, elem):
        """
        Return the number of elements equal to elem present in the queue

        >>> pdeque([1, 2, 1]).count(1)
        2
        """
        return (sum(1 for e in _stream_iter(self._left_stream) if e == elem) +
                sum(1 for e in _stream_iter(self._right_stream) if e == elem))

    def remove(self, elem):
        """
        Return new deque with first element from left equal to elem removed. If no such element is found
        a ValueError is raised.

        >>> pdeque([2, 1, 2]).remove(2)
        pdeque([1, 2])
        """
        left = _realized(self._left_stream)
        try:
            new_left = _stream_remove_first(left, elem)
            return self._from_state(
                (new_left, _EMPTY_STREAM, self._left_length - 1,
                 _realized(self._right_stream), _EMPTY_STREAM, self._right_length),
                self._maxlen)
        except ValueError:
            pass

        right = _realized(self._right_stream)
        try:
            new_right = _stream_reverse(_stream_remove_first(_stream_reverse(right), elem))
        except ValueError as e:
            raise ValueError('{0} not found in PDeque'.format(elem)) from e

        return self._from_state(
            (_realized(self._left_stream), _EMPTY_STREAM, self._left_length,
             new_right, _EMPTY_STREAM, self._right_length - 1),
            self._maxlen)

    def reverse(self):
        """
        Return reversed deque.

        >>> pdeque([1, 2, 3]).reverse()
        pdeque([3, 2, 1])

        Also supports the standard python reverse function.

        >>> reversed(pdeque([1, 2, 3]))
        pdeque([3, 2, 1])
        """
        return PDeque(self._right_stream, self._left_stream,
                      self._right_length, self._left_length, self._maxlen,
                      left_sched=self._right_sched, right_sched=self._left_sched)
    __reversed__ = reverse

    def rotate(self, steps):
        """
        Return deque with elements rotated steps steps.

        >>> x = pdeque([1, 2, 3])
        >>> x.rotate(1)
        pdeque([3, 1, 2])
        >>> x.rotate(-2)
        pdeque([3, 1, 2])
        """
        popped_deque = self.pop(steps)
        if steps >= 0:
            return popped_deque.extendleft(islice(self.reverse(), steps))

        return popped_deque.extend(islice(self, -steps))

    def __reduce__(self):
        # Pickling support
        return pdeque, (list(self), self._maxlen)

    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.step is not None and index.step != 1:
                # Too difficult, no structural sharing possible
                return pdeque(tuple(self)[index], maxlen=self._maxlen)

            length = len(self)
            result = self
            if index.start is not None:
                result = result.popleft(index.start % length)
            if index.stop is not None:
                result = result.pop(length - (index.stop % length))

            return result

        if not isinstance(index, Integral):
            raise TypeError("'%s' object cannot be interpreted as an index" % type(index).__name__)

        if index >= 0:
            return self.popleft(index).left

        shifted = len(self) + index
        if shifted < 0:
            raise IndexError(
                "pdeque index {0} out of range {1}".format(index, len(self)),
            )
        return self.popleft(shifted).left

    index = Sequence.index

Sequence.register(PDeque)
Hashable.register(PDeque)


def pdeque(iterable=(), maxlen=None):
    """
    Return deque containing the elements of iterable. If maxlen is specified then
    len(iterable) - maxlen elements are discarded from the left to if len(iterable) > maxlen.

    >>> pdeque([1, 2, 3])
    pdeque([1, 2, 3])
    >>> pdeque([1, 2, 3, 4], maxlen=2)
    pdeque([3, 4], maxlen=2)
    """
    t = tuple(iterable)
    if maxlen is not None and len(t) > maxlen:
        t = t[len(t) - maxlen:]

    length = len(t)
    pivot = length // 2

    left = _EMPTY_STREAM
    for elem in reversed(t[:pivot]):
        left = _Stream.cons(elem, left)

    right = _EMPTY_STREAM
    for elem in t[pivot:]:
        right = _Stream.cons(elem, right)

    return PDeque(left, right, pivot, length - pivot, maxlen)

def dq(*elements):
    """
    Return deque containing all arguments.

    >>> dq(1, 2, 3)
    pdeque([1, 2, 3])
    """
    return pdeque(elements)

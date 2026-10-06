# -*- coding: utf-8 -*-

"""
Persistent (immutable) priority queue with stable element handles.

Every operation returns a new :class:`PPriorityQueue` version; old versions
stay valid and unchanged, so backtracking search can keep a whole tree of
branches alive and let each branch evolve independently.  Unchanged nodes
are shared structurally between versions (path copying), no wholesale list
copying like a naive ``heapq`` snapshot would require.

Elements are inserted together with a priority and are dequeued smallest
priority first.  Ties are broken by a stable declaration order: the earlier
an element was first inserted into the version lineage, the earlier it is
dequeued.  (Two branches may independently declare different elements under
the same sequence number; if such elements meet in a later merge with equal
priority, the tie is broken deterministically by handle creation order.)
Insertion returns a :class:`Handle` which stays valid across all
versions derived from the insertion point and can be used to reprioritize
(:meth:`PPriorityQueue.set_priority`) or inspect the element.

Two versions derived from a common ancestor can be recombined with
:meth:`PPriorityQueue.merge`.  Every entry carries a write epoch, so the
merge can tell a stale ancestor copy apart from a genuinely diverging
concurrent write: the fresher entry wins and the divergence is reported.
If both branches rewrote the same handle concurrently (same epoch,
different content) the conflict is settled through an explicit policy
(``'error'``, ``'left'``, ``'right'`` or a callable).  Either way a
diverging handle is never silently duplicated in the merged queue.

Handles that are unknown to a version (never inserted, already popped, or
belonging to an unrelated queue) are rejected with :class:`InvalidHandleError`.
"""

from collections import namedtuple
from itertools import count


__all__ = ('Handle', 'PPriorityQueue', 'ppriorityqueue', 'Item',
           'MergeConflict', 'InvalidHandleError', 'MergeConflictError')


# ---------------------------------------------------------------------------
# Immutable AVL tree (ordered map) with path copying.
#
# Every update creates only the O(log n) nodes on the affected path; all
# other nodes are shared with the previous version.
# ---------------------------------------------------------------------------

class _AVLNode(object):
    __slots__ = ('key', 'value', 'left', 'right', 'height', 'size')

    def __init__(self, key, value, left, right):
        self.key = key
        self.value = value
        self.left = left
        self.right = right
        self.height = 1 + max(_height(left), _height(right))
        self.size = 1 + _size(left) + _size(right)


def _height(node):
    return node.height if node is not None else 0


def _size(node):
    return node.size if node is not None else 0


def _rotate_left(node):
    pivot = node.right
    new_left = _AVLNode(node.key, node.value, node.left, pivot.left)
    return _AVLNode(pivot.key, pivot.value, new_left, pivot.right)


def _rotate_right(node):
    pivot = node.left
    new_right = _AVLNode(node.key, node.value, pivot.right, node.right)
    return _AVLNode(pivot.key, pivot.value, pivot.left, new_right)


def _rebalance(node):
    balance = _height(node.left) - _height(node.right)
    if balance > 1:
        left = node.left
        if _height(left.left) < _height(left.right):
            left = _rotate_left(left)
        return _rotate_right(_AVLNode(node.key, node.value, left, node.right))
    if balance < -1:
        right = node.right
        if _height(right.right) < _height(right.left):
            right = _rotate_right(right)
        return _rotate_left(_AVLNode(node.key, node.value, node.left, right))
    return node


def _insert(node, key, value):
    if node is None:
        return _AVLNode(key, value, None, None)
    if key < node.key:
        return _rebalance(_AVLNode(node.key, node.value,
                                   _insert(node.left, key, value), node.right))
    if key > node.key:
        return _rebalance(_AVLNode(node.key, node.value, node.left,
                                   _insert(node.right, key, value)))
    return _AVLNode(key, value, node.left, node.right)


def _remove(node, key):
    if node is None:
        raise KeyError(key)
    if key < node.key:
        return _rebalance(_AVLNode(node.key, node.value,
                                   _remove(node.left, key), node.right))
    if key > node.key:
        return _rebalance(_AVLNode(node.key, node.value, node.left,
                                   _remove(node.right, key)))
    if node.left is None:
        return node.right
    if node.right is None:
        return node.left
    succ_key, succ_value = _min_pair(node.right)
    return _rebalance(_AVLNode(succ_key, succ_value, node.left,
                               _remove(node.right, succ_key)))


def _get(node, key, default=None):
    while node is not None:
        if key < node.key:
            node = node.left
        elif key > node.key:
            node = node.right
        else:
            return node.value
    return default


def _min_pair(node):
    if node is None:
        raise KeyError('empty tree')
    while node.left is not None:
        node = node.left
    return node.key, node.value


def _items(node):
    """In-order iteration over (key, value) pairs."""
    stack = []
    while stack or node is not None:
        while node is not None:
            stack.append(node)
            node = node.left
        node = stack.pop()
        yield node.key, node.value
        node = node.right


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

_handle_uids = count()


class Handle(object):
    """
    Stable identity of one queue element.

    A handle is created by :meth:`PPriorityQueue.insert` and remains valid in
    every version derived from the version that created it.  Handles compare
    and hash by identity, so they can be used as dictionary keys.
    """
    __slots__ = ('_uid',)

    def __init__(self):
        self._uid = next(_handle_uids)

    @property
    def uid(self):
        return self._uid

    def __eq__(self, other):
        return isinstance(other, Handle) and self._uid == other._uid

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._uid)

    def __repr__(self):
        return 'Handle(%d)' % (self._uid,)


class InvalidHandleError(KeyError):
    """Raised when a handle is unknown to (or no longer present in) a version."""


class MergeConflictError(Exception):
    """
    Raised by :meth:`PPriorityQueue.merge` with policy ``'error'`` when both
    branches carry diverging versions of the same handle.

    The ``conflicts`` attribute lists every conflicting handle as a
    :class:`MergeConflict` named tuple.
    """

    def __init__(self, conflicts):
        self.conflicts = list(conflicts)
        super(MergeConflictError, self).__init__(
            '%d merge conflict(s) over handle(s) %s'
            % (len(self.conflicts), [c.handle for c in self.conflicts]))


#: A dequeued / inspected element.
Item = namedtuple('Item', ('handle', 'payload', 'priority'))

#: One merge conflict: both sides of the divergence as (priority, payload)
#: pairs plus the resolution that was kept (None when policy is 'error').
MergeConflict = namedtuple('MergeConflict', ('handle', 'left', 'right', 'resolution'))

# Internal per-handle record.  ``epoch`` is the value of the queue's write
# counter when this entry was last (re)written; it lets merge distinguish a
# stale ancestor copy from a concurrent diverging write.
_Entry = namedtuple('_Entry', ('handle', 'priority', 'seq', 'epoch', 'payload'))


class PPriorityQueue(object):
    """
    Immutable priority queue.  All mutating operations return new versions.

    Supports ``insert``, ``pop_min``/``peek_min``, ``set_priority`` on known
    handles and ``merge`` of two versions with an explicit conflict policy.
    All single-version operations are O(log n); versions derived from each
    other share all untouched nodes.
    """
    __slots__ = ('_entries', '_order', '_next_seq', '_epoch')

    def __init__(self, entries=None, order=None, next_seq=0, epoch=0):
        # _entries: AVL mapping handle uid -> _Entry
        # _order:   AVL mapping (priority, seq, handle uid) -> handle uid
        self._entries = entries
        self._order = order
        self._next_seq = next_seq
        self._epoch = epoch

    # -- queries -----------------------------------------------------------

    def __len__(self):
        return _size(self._entries)

    def __contains__(self, handle):
        return (isinstance(handle, Handle)
                and _get(self._entries, handle._uid) is not None)

    def __iter__(self):
        """Yields :class:`Item` objects in dequeue order, non-destructively."""
        for _, uid in _items(self._order):
            entry = _get(self._entries, uid)
            yield Item(entry.handle, entry.payload, entry.priority)

    def __eq__(self, other):
        return (isinstance(other, PPriorityQueue)
                and list(_items(self._entries)) == list(_items(other._entries)))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'PPriorityQueue(%s)' % (list(self),)

    def _entry_for(self, handle):
        if not isinstance(handle, Handle):
            raise InvalidHandleError(handle)
        entry = _get(self._entries, handle._uid)
        if entry is None:
            raise InvalidHandleError(handle)
        return entry

    def get(self, handle):
        """Returns the :class:`Item` for *handle*; rejects unknown handles."""
        entry = self._entry_for(handle)
        return Item(entry.handle, entry.payload, entry.priority)

    def peek_min(self):
        """Returns the minimal :class:`Item` without modifying the queue."""
        if self._order is None:
            raise IndexError('peek_min from empty priority queue')
        _, uid = _min_pair(self._order)
        entry = _get(self._entries, uid)
        return Item(entry.handle, entry.payload, entry.priority)

    # -- derived versions --------------------------------------------------

    def insert(self, payload, priority=0):
        """
        Returns ``(handle, new_queue)`` with *payload* stored at *priority*.

        The element's declaration sequence number is fixed here and used as
        the stable tie-breaker between equal priorities in every descendant
        version.
        """
        handle = Handle()
        seq = self._next_seq
        epoch = self._epoch + 1
        entry = _Entry(handle, priority, seq, epoch, payload)
        entries = _insert(self._entries, handle._uid, entry)
        order = _insert(self._order, (priority, seq, handle._uid), handle._uid)
        return handle, PPriorityQueue(entries, order, seq + 1, epoch)

    def set_priority(self, handle, priority):
        """
        Returns a new version with *handle* reprioritized.

        The element keeps its declaration sequence number, so its relative
        order among elements with the same new priority is preserved.
        """
        entry = self._entry_for(handle)
        if entry.priority == priority:
            return self
        epoch = self._epoch + 1
        order = _remove(self._order, (entry.priority, entry.seq, handle._uid))
        order = _insert(order, (priority, entry.seq, handle._uid), handle._uid)
        entries = _insert(self._entries, handle._uid,
                          _Entry(handle, priority, entry.seq, epoch,
                                 entry.payload))
        return PPriorityQueue(entries, order, self._next_seq, epoch)

    def pop_min(self):
        """
        Returns ``(item, new_queue)`` where *item* is the minimal element
        (by priority, then declaration order) and *new_queue* lacks it.

        The popped handle becomes invalid in the new version but stays valid
        in this (unchanged) version.
        """
        if self._order is None:
            raise IndexError('pop_min from empty priority queue')
        key, uid = _min_pair(self._order)
        entry = _get(self._entries, uid)
        order = _remove(self._order, key)
        entries = _remove(self._entries, uid)
        item = Item(entry.handle, entry.payload, entry.priority)
        return item, PPriorityQueue(entries, order, self._next_seq, self._epoch)

    def merge(self, other, on_conflict='error'):
        """
        Merges two versions into one, returning ``(new_queue, conflicts)``.

        Handles present in only one version are carried over.  Handles both
        versions agree on appear exactly once.  When the two versions carry
        different entries for the same handle, their write epochs decide:

        * different epochs -- the older entry is a stale ancestor copy (or
          an earlier write); the fresher entry wins automatically
        * same epoch -- both branches rewrote the handle concurrently, a
          genuine conflict that is settled by *on_conflict*

        Every divergence (automatic or policy-resolved) is reported as a
        :class:`MergeConflict` in the returned list, so a diverging handle
        is never silently dequeued twice.  *on_conflict* may be:

        * ``'error'`` -- raise :class:`MergeConflictError` listing all of them
        * ``'left'`` / ``'right'`` -- keep this queue's / *other*'s version
        * a callable ``f(handle, left, right) -> (priority, payload)`` where
          *left* and *right* are ``(priority, payload)`` pairs

        Merge is set-union over live entries: popping a handle in one branch
        does not remove the other branch's copy.  It runs in O(n log n) in
        the size of the result.
        """
        if not isinstance(other, PPriorityQueue):
            raise TypeError('can only merge with another PPriorityQueue, '
                            'got %r' % (type(other),))
        resolver = _conflict_resolver(on_conflict)

        conflicts = []
        merged_entries = []
        sentinel = object()
        left_iter = _items(self._entries)
        right_iter = _items(other._entries)
        left = next(left_iter, sentinel)
        right = next(right_iter, sentinel)
        while left is not sentinel or right is not sentinel:
            if right is sentinel or (left is not sentinel and left[0] < right[0]):
                merged_entries.append(left[1])
                left = next(left_iter, sentinel)
            elif left is sentinel or right[0] < left[0]:
                merged_entries.append(right[1])
                right = next(right_iter, sentinel)
            else:
                left_entry, right_entry = left[1], right[1]
                left_side = (left_entry.priority, left_entry.payload)
                right_side = (right_entry.priority, right_entry.payload)
                if left_side == right_side:
                    # Same content; keep the fresher write stamp.
                    merged_entries.append(left_entry
                                          if left_entry.epoch >= right_entry.epoch
                                          else right_entry)
                elif left_entry.epoch != right_entry.epoch:
                    # One side is a stale ancestor copy (or earlier write);
                    # the fresher write wins, the divergence is reported.
                    winner = (left_entry if left_entry.epoch > right_entry.epoch
                              else right_entry)
                    merged_entries.append(winner)
                    conflicts.append(MergeConflict(
                        left_entry.handle, left_side, right_side,
                        (winner.priority, winner.payload)))
                else:
                    # Genuine concurrent rewrite of the same handle.
                    if resolver is None:
                        resolution = None
                        merged_entries.append(left_entry)  # discarded by raise
                    else:
                        resolution = resolver(left_entry.handle, left_side, right_side)
                        _validate_resolution(resolution)
                        merged_entries.append(_Entry(left_entry.handle,
                                                     resolution[0],
                                                     left_entry.seq,
                                                     left_entry.epoch,
                                                     resolution[1]))
                    conflicts.append(MergeConflict(left_entry.handle, left_side,
                                                   right_side, resolution))
                left = next(left_iter, sentinel)
                right = next(right_iter, sentinel)

        if conflicts and resolver is None:
            raise MergeConflictError(conflicts)

        entries = None
        order = None
        for entry in merged_entries:
            entries = _insert(entries, entry.handle._uid, entry)
            order = _insert(order,
                            (entry.priority, entry.seq, entry.handle._uid),
                            entry.handle._uid)
        queue = PPriorityQueue(entries, order,
                               max(self._next_seq, other._next_seq),
                               max(self._epoch, other._epoch))
        return queue, conflicts


def _conflict_resolver(on_conflict):
    if on_conflict == 'error':
        return None
    if on_conflict == 'left':
        return lambda handle, left, right: left
    if on_conflict == 'right':
        return lambda handle, left, right: right
    if callable(on_conflict):
        return on_conflict
    raise ValueError("on_conflict must be 'error', 'left', 'right' or a "
                     'callable, got %r' % (on_conflict,))


def _validate_resolution(resolution):
    if (not isinstance(resolution, tuple) or len(resolution) != 2):
        raise TypeError('conflict resolver must return a (priority, payload) '
                        'tuple, got %r' % (resolution,))


_EMPTY_QUEUE = PPriorityQueue()


def ppriorityqueue():
    """Returns the empty (singleton) persistent priority queue."""
    return _EMPTY_QUEUE

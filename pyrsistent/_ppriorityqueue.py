from __future__ import annotations

from collections.abc import Iterable, Iterator, Sized
from enum import Enum
from itertools import count
from typing import Any, NamedTuple, Optional
from uuid import uuid4

from pyrsistent._pmap import pmap


_MISSING = object()


class PriorityMergePolicy(str, Enum):
    REJECT = 'reject'
    LEFT = 'left'
    RIGHT = 'right'
    MIN_PRIORITY = 'min_priority'


class PQHandle(NamedTuple):
    family_id: int
    item_id: int

    def __repr__(self):
        return f'PQHandle(family={self.family_id}, item={self.item_id})'


class PQVersion(NamedTuple):
    revision: int
    parents: tuple[PQVersion, ...]


class PQItem(NamedTuple):
    handle: PQHandle
    value: Any
    priority: Any
    tie: int


class PQMergeConflict(NamedTuple):
    handle: PQHandle
    left: Optional[PQItem]
    right: Optional[PQItem]


class PPriorityQueueError(Exception):
    """Base class for persistent priority queue errors."""


class PPriorityQueueHandleError(PPriorityQueueError, KeyError):
    """Raised when a priority queue handle is invalid."""


class PPriorityQueueEmptyError(PPriorityQueueError, IndexError):
    """Raised when popping an empty persistent priority queue."""


class PPriorityQueueMergeError(PPriorityQueueError, ValueError):
    """Raised when two priority queues cannot be merged."""


class PPriorityQueueConflictError(PPriorityQueueMergeError):
    """Raised when merging divergent handle versions without a resolution."""

    def __init__(self, message, conflicts):
        super().__init__(message)
        self.conflicts = tuple(conflicts)


class _Family:
    __slots__ = ('family_id', 'items')

    def __init__(self):
        self.family_id = uuid4().int
        self.items = count()

    def revision(self, parents):
        return PQVersion(uuid4().int, tuple(parents))


class _Entry(NamedTuple):
    value: Any
    priority: Any
    tie: int
    version: PQVersion


class _Tombstone(NamedTuple):
    version: PQVersion


class _TreapNode(NamedTuple):
    key: tuple[Any, int]
    entry: _Entry
    left: Optional['_TreapNode']
    right: Optional['_TreapNode']


def _rank(value):
    value = (value + 0x9e3779b97f4a7c15) & 0xffffffffffffffff
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & 0xffffffffffffffff
    value = ((value ^ (value >> 27)) * 0x1b4856a3df71bd77) & 0xffffffffffffffff
    return value ^ (value >> 31)


def _treap_insert(node, key, entry):
    if node is None:
        return _TreapNode(key, entry, None, None)
    if key < node.key:
        left = _treap_insert(node.left, key, entry)
        node = _TreapNode(node.key, node.entry, left, node.right)
        if _rank(left.key[1]) > _rank(node.key[1]):
            return _TreapNode(
                left.key, left.entry, left.left,
                _TreapNode(node.key, node.entry, left.right, node.right))
    elif key > node.key:
        right = _treap_insert(node.right, key, entry)
        node = _TreapNode(node.key, node.entry, node.left, right)
        if _rank(right.key[1]) > _rank(node.key[1]):
            return _TreapNode(
                right.key, right.entry,
                _TreapNode(node.key, node.entry, node.left, right.left),
                right.right)
    return node


def _treap_remove(node, key):
    if node is None:
        return None
    if key < node.key:
        return _TreapNode(node.key, node.entry,
                          _treap_remove(node.left, key), node.right)
    if key > node.key:
        return _TreapNode(node.key, node.entry, node.left,
                          _treap_remove(node.right, key))
    return _treap_union(node.left, node.right)


def _treap_replace(node, key, entry):
    if node is None:
        raise KeyError(key)
    if key < node.key:
        return _TreapNode(node.key, node.entry,
                          _treap_replace(node.left, key, entry), node.right)
    if key > node.key:
        return _TreapNode(node.key, node.entry, node.left,
                          _treap_replace(node.right, key, entry))
    return _TreapNode(node.key, entry, node.left, node.right)


def _treap_union(left, right):
    if left is None:
        return right
    if right is None:
        return left
    if left is right:
        return left

    left_keys = _treap_key_set(left)
    added_keys = [
        node.key for node in _treap_items(right)
        if node.key not in left_keys
    ]
    merged = left
    for key in added_keys:
        entry = _treap_find(right, key)
        merged = _treap_insert(merged, key, entry)
    return merged


def _treap_key_set(node):
    return {item.key for item in _treap_items(node)}


def _treap_find(node, key):
    while node is not None:
        if key < node.key:
            node = node.left
        elif key > node.key:
            node = node.right
        else:
            return node.entry
    raise KeyError(key)


def _treap_min(node):
    while node.left is not None:
        node = node.left
    return node


def _treap_items(node):
    if node is None:
        return
    yield from _treap_items(node.left)
    yield node
    yield from _treap_items(node.right)


class PPriorityQueue(Sized):
    """
    Immutable priority queue with stable, version-safe item handles.

    Equal priorities are popped in insertion order. Updates return a new queue
    without changing old branches. Merges preserve one version of each handle
    and use ``PriorityMergePolicy`` to report or resolve divergent versions.
    """
    __slots__ = ('_family', '_heap', '_active', '_history', '_version', '_size')

    def __new__(cls, family=None, heap=None, active=None, history=None,
                version=None, size=0):
        queue = super().__new__(cls)
        object.__setattr__(queue, '_family', family or _Family())
        object.__setattr__(queue, '_heap', heap)
        object.__setattr__(queue, '_active', pmap() if active is None else active)
        object.__setattr__(queue, '_history', pmap() if history is None else history)
        object.__setattr__(queue, '_version',
                           version or queue._family.revision(()))
        object.__setattr__(queue, '_size', size)
        return queue

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def __iter__(self) -> Iterator[PQItem]:
        for node in _treap_items(self._heap):
            entry = self._active.get(node.key[1], _MISSING)
            if entry is not None and entry is node.entry and \
                    node.key[0] == entry.priority:
                yield PQItem(PQHandle(self._family.family_id, node.key[1]),
                             entry.value, entry.priority, entry.tie)

    def __contains__(self, handle):
        self._validate_handle(handle)
        return handle.item_id in self._active

    def __eq__(self, other):
        if not isinstance(other, PPriorityQueue):
            return NotImplemented
        if self._family is not other._family:
            return False
        return list(self) == list(other)

    def __repr__(self):
        return f'ppriorityqueue({[(item.value, item.priority) for item in self]})'

    def copy(self):
        return self

    def insert(self, value, priority):
        item_id = next(self._family.items)
        tie = item_id
        version = self._family.revision((self._version,))
        entry = _Entry(value, priority, tie, version)
        key = (priority, tie)
        handle = PQHandle(self._family.family_id, item_id)
        heap = _treap_insert(self._heap, key, entry)
        active = self._active.set(item_id, entry)
        history = self._history.set(item_id, entry)
        return PPriorityQueue(self._family, heap, active, history,
                              version, self._size + 1), handle

    def push(self, value, priority):
        return self.insert(value, priority)

    def update_priority(self, handle, priority):
        old = self._validate_active_handle(handle)
        if priority == old.priority:
            return self
        return self._replace_entry(handle, old, priority, old.value)

    def set_value(self, handle, value):
        old = self._validate_active_handle(handle)
        return self._replace_entry(handle, old, old.priority, value)

    def _replace_entry(self, handle, old, priority, value):
        entry_version = self._family.revision((old.version,))
        queue_version = self._family.revision((self._version,))
        entry = _Entry(value, priority, old.tie, entry_version)
        old_key = (old.priority, old.tie)
        new_key = (priority, old.tie)
        if new_key == old_key:
            heap = _treap_replace(self._heap, old_key, entry)
        else:
            heap = _treap_remove(self._heap, old_key)
            heap = _treap_insert(heap, new_key, entry)
        active = self._active.set(handle.item_id, entry)
        history = self._history.set(handle.item_id, entry)
        return PPriorityQueue(self._family, heap, active, history,
                              queue_version, self._size)

    def peek_min(self):
        node = self._min_node()
        if node is None:
            raise PPriorityQueueEmptyError('pop from empty PPriorityQueue')
        entry = self._active[node.key[1]]
        return PQItem(PQHandle(self._family.family_id, node.key[1]),
                      entry.value, entry.priority, entry.tie)

    def pop_min(self):
        node = self._min_node()
        if node is None:
            raise PPriorityQueueEmptyError('pop from empty PPriorityQueue')
        handle = PQHandle(self._family.family_id, node.key[1])
        old = self._active[node.key[1]]
        record_version = self._family.revision((old.version,))
        queue_version = self._family.revision((self._version,))
        tombstone = _Tombstone(record_version)
        heap = _treap_remove(self._heap, node.key)
        active = self._active.remove(node.key[1])
        history = self._history.set(node.key[1], tombstone)
        item = PQItem(handle, old.value, old.priority, old.tie)
        return PPriorityQueue(self._family, heap, active, history,
                              queue_version, self._size - 1), item

    def _min_node(self):
        for node in _treap_items(self._heap):
            entry = self._active.get(node.key[1], _MISSING)
            if entry is node.entry and node.key[0] == entry.priority:
                return node
        return None

    def merge(self, other, policy=PriorityMergePolicy.REJECT):
        if not isinstance(other, PPriorityQueue):
            raise PPriorityQueueMergeError('can only merge PPriorityQueue instances')
        if self._family is not other._family:
            raise PPriorityQueueMergeError('queues belong to different queue families')

        policy = self._merge_policy(policy)
        heap = _treap_union(self._heap, other._heap)
        active = self._active.evolver()
        history = self._history.evolver()
        conflicts = []
        actions = []
        other_is_descendant = self._is_version_of(self._version, other._version)
        self_is_descendant = self._is_version_of(other._version, self._version)

        for handle_id in self._history.keys() | other._history.keys():
            left = self._history.get(handle_id)
            right = other._history.get(handle_id)
            relation = self._record_relation(left, right)

            if relation == 'equal':
                continue
            if relation == 'left':
                if isinstance(right, _Entry) and (
                        not isinstance(left, _Entry) or
                        (right.priority, right.tie) != (left.priority, left.tie)):
                    heap = _treap_remove(heap, (right.priority, right.tie))
                continue
            if relation == 'right':
                if isinstance(left, _Entry) and (
                        not isinstance(right, _Entry) or
                        (left.priority, left.tie) != (right.priority, right.tie)):
                    heap = _treap_remove(heap, (left.priority, left.tie))
                actions.append(('use', handle_id, left, right, right))
                continue
            if left is None or right is None:
                record = right if left is None else left
                actions.append(('use', handle_id, left, right, record))
                continue

            left_active = isinstance(left, _Entry)
            right_active = isinstance(right, _Entry)
            if not left_active and not right_active:
                continue

            conflict = PQMergeConflict(
                PQHandle(self._family.family_id, handle_id),
                self._item_from_record(handle_id, left) if left_active else None,
                self._item_from_record(handle_id, right) if right_active else None)
            if policy == PriorityMergePolicy.REJECT:
                conflicts.append(conflict)
                continue

            selected = self._select_conflict(left, right, policy, conflict)
            if left_active:
                heap = _treap_remove(heap, (left.priority, left.tie))
            if right_active:
                heap = _treap_remove(heap, (right.priority, right.tie))
            if handle_id in active:
                del active[handle_id]
            actions.append(('divergent', handle_id, left, right, selected))

        if conflicts:
            raise PPriorityQueueConflictError(
                'conflicting versions while merging priority queues', conflicts)

        if not actions:
            return self
        if other_is_descendant and all(action[0] == 'use' and action[4] is action[3]
               for action in actions):
            return other
        if self_is_descendant and all(action[0] == 'use' and action[4] is action[2]
               for action in actions):
            return self

        merged_version = self._family.revision((self._version, other._version))
        for action in actions:
            if action[0] == 'use':
                _, handle_id, left, right, record = action
                heap = self._apply_record(
                    heap, active, history, handle_id, left, right,
                    record)
            else:
                _, handle_id, left, right, selected = action
                heap = self._apply_divergent_record(
                    heap, active, history, handle_id, left, right,
                    selected)

        return PPriorityQueue(self._family, heap, active.persistent(),
                              history.persistent(), merged_version, len(active))

    def _apply_record(self, heap, active, history, handle_id, left, right, record):
        if left is not None and isinstance(left, _Entry):
            heap = _treap_remove(heap, (left.priority, left.tie))
        if right is not None and isinstance(right, _Entry):
            heap = _treap_remove(heap, (right.priority, right.tie))
        if isinstance(record, _Entry):
            active[handle_id] = record
            history[handle_id] = record
            return _treap_insert(heap, (record.priority, record.tie), record)

        if handle_id in active:
            del active[handle_id]
        history[handle_id] = record
        return heap

    def _apply_divergent_record(self, heap, active, history, handle_id,
                                left, right, selected):
        left_active = isinstance(left, _Entry)
        right_active = isinstance(right, _Entry)

        if selected is None:
            if handle_id in active:
                del active[handle_id]
            history[handle_id] = left if isinstance(left, _Tombstone) else right
            return heap

        heap = _treap_insert(heap, (selected.priority, selected.tie), selected)
        active[handle_id] = selected
        history[handle_id] = selected
        return heap

    def _select_conflict(self, left, right, policy, conflict):
        left_active = isinstance(left, _Entry)
        right_active = isinstance(right, _Entry)
        if policy == PriorityMergePolicy.LEFT:
            return left if left_active else None
        if policy == PriorityMergePolicy.RIGHT:
            return right if right_active else None
        if policy == PriorityMergePolicy.MIN_PRIORITY and left_active and right_active:
            return left if left.priority <= right.priority else right
        if policy == PriorityMergePolicy.MIN_PRIORITY:
            raise PPriorityQueueConflictError(
                'min_priority cannot resolve insertion versus deletion', [conflict])
        raise PPriorityQueueConflictError(
            f'unknown merge policy: {policy!r}', [conflict])

    @staticmethod
    def _merge_policy(policy):
        if isinstance(policy, PriorityMergePolicy):
            return policy.value
        try:
            return PriorityMergePolicy(policy).value
        except ValueError as error:
            message = f'unknown merge policy: {policy!r}'
            raise PPriorityQueueMergeError(message) from error

    def _record_relation(self, left, right):
        if left is None and right is None:
            return 'equal'
        if left is None:
            return 'right'
        if right is None:
            return 'left'
        if left.version == right.version:
            return 'equal'
        if self._is_version_of(left.version, right.version):
            return 'right'
        if self._is_version_of(right.version, left.version):
            return 'left'
        return 'divergent'

    @staticmethod
    def _is_version_of(old, new):
        if old == new:
            return True
        seen = set()
        pending = list(new.parents)
        while pending:
            version = pending.pop()
            if version in seen:
                continue
            seen.add(version)
            if version == old:
                return True
            pending.extend(version.parents)
        return False

    def _item_from_record(self, handle_id, record):
        return PQItem(PQHandle(self._family.family_id, handle_id),
                      record.value, record.priority, record.tie)

    def _validate_handle(self, handle):
        if not isinstance(handle, PQHandle):
            message = f'invalid priority queue handle: {handle!r}'
            raise PPriorityQueueHandleError(message)
        if handle.family_id != self._family.family_id:
            raise PPriorityQueueHandleError('handle belongs to another priority queue')

    def _validate_active_handle(self, handle):
        self._validate_handle(handle)
        try:
            record = self._active[handle.item_id]
        except KeyError:
            if handle.item_id in self._history:
                raise PPriorityQueueHandleError(
                    f'handle {handle.item_id!r} has already been popped') from None
            raise PPriorityQueueHandleError(
                f'unknown priority queue handle: {handle.item_id!r}') from None
        return record


def ppriorityqueue(iterable=()):
    queue = PPriorityQueue()
    for item in iterable:
        if isinstance(item, tuple) and len(item) == 2:
            value, priority = item
        else:
            value, priority = item
        queue, _ = queue.insert(value, priority)
    return queue


def pq(*items):
    if len(items) == 1 and not (isinstance(items[0], tuple) and len(items[0]) == 2):
        return ppriorityqueue(items[0])
    return ppriorityqueue(items)

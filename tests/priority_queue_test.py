# -*- coding: utf-8 -*-
import random

import pytest

from pyrsistent import (ppriorityqueue, PPriorityQueue, Handle, MergeConflict,
                        InvalidHandleError, MergeConflictError)


class Reference(object):
    """Plain mutable dict model used as ground truth for one version."""

    def __init__(self):
        self.entries = {}  # handle -> (priority, seq, epoch, payload)
        self.next_seq = 0
        self.epoch = 0

    def clone(self):
        other = Reference()
        other.entries = dict(self.entries)
        other.next_seq = self.next_seq
        other.epoch = self.epoch
        return other

    def insert(self, handle, payload, priority):
        self.epoch += 1
        self.entries[handle] = (priority, self.next_seq, self.epoch, payload)
        self.next_seq += 1

    def set_priority(self, handle, priority):
        old_priority, seq, _, payload = self.entries[handle]
        if priority == old_priority:
            return
        self.epoch += 1
        self.entries[handle] = (priority, seq, self.epoch, payload)

    def pop_min(self):
        handle, (priority, seq, _, payload) = min(
            self.entries.items(),
            key=lambda kv: (kv[1][0], kv[1][1], kv[0].uid))
        del self.entries[handle]
        return handle, payload, priority

    def ordered(self):
        return [(handle, payload, priority)
                for handle, (priority, seq, _, payload) in sorted(
                    self.entries.items(),
                    key=lambda kv: (kv[1][0], kv[1][1], kv[0].uid))]

    def merge(self, other, policy):
        out = self.clone()
        out.next_seq = max(self.next_seq, other.next_seq)
        out.epoch = max(self.epoch, other.epoch)
        for handle, entry in other.entries.items():
            mine = out.entries.get(handle)
            if mine is None:
                out.entries[handle] = entry
            elif (mine[0], mine[3]) == (entry[0], entry[3]):
                out.entries[handle] = max(mine, entry, key=lambda e: e[2])
            elif mine[2] != entry[2]:
                out.entries[handle] = max(mine, entry, key=lambda e: e[2])
            elif policy == 'right':
                out.entries[handle] = entry
        return out


def assert_matches(queue, model):
    """The queue must yield exactly the model's elements, in pop order."""
    assert [(item.handle, item.payload, item.priority) for item in queue] \
        == model.ordered()
    assert len(queue) == len(model.entries)
    # Drain a copy via pop_min; the drained version must follow the same order.
    drained = []
    while len(queue):
        item, queue = queue.pop_min()
        drained.append((item.handle, item.payload, item.priority))
    assert drained == model.ordered()


def test_empty_queue():
    queue = ppriorityqueue()
    assert len(queue) == 0
    assert list(queue) == []
    with pytest.raises(IndexError):
        queue.pop_min()
    with pytest.raises(IndexError):
        queue.peek_min()


def test_pop_order_is_priority_then_declaration_order():
    queue = ppriorityqueue()
    for payload, priority in [('x', 2), ('a', 1), ('m', 2), ('b', 1), ('z', 5)]:
        _, queue = queue.insert(payload, priority)
    drained = []
    while len(queue):
        item, queue = queue.pop_min()
        drained.append(item.payload)
    assert drained == ['a', 'b', 'x', 'm', 'z']


def test_set_priority_keeps_stable_order_and_old_version():
    queue = ppriorityqueue()
    ha, queue = queue.insert('a', 1)
    hb, queue = queue.insert('b', 1)
    hc, queue = queue.insert('c', 1)

    promoted = queue.set_priority(hc, 0)
    assert [item.payload for item in promoted] == ['c', 'a', 'b']
    # The original version is untouched.
    assert [item.payload for item in queue] == ['a', 'b', 'c']
    # Reprioritizing to the same value is a no-op.
    assert queue.set_priority(ha, 1) is queue
    # Ties after reprioritization still follow declaration order.
    demoted = queue.set_priority(ha, 2).set_priority(hb, 2)
    assert [item.payload for item in demoted] == ['c', 'a', 'b']


def test_foreign_stale_and_non_handles_are_rejected():
    handle, queue = ppriorityqueue().insert('a', 1)
    foreign, other_queue = ppriorityqueue().insert('b', 1)

    with pytest.raises(InvalidHandleError):
        queue.get(foreign)
    with pytest.raises(InvalidHandleError):
        queue.set_priority(foreign, 5)
    with pytest.raises(InvalidHandleError):
        queue.get('not a handle')
    with pytest.raises(InvalidHandleError):
        queue.get(None)

    item, popped = queue.pop_min()
    assert item.handle == handle
    with pytest.raises(InvalidHandleError):
        popped.get(handle)
    assert handle not in popped
    # ...but the handle stays valid in the pre-pop version.
    assert queue.get(handle).payload == 'a'
    assert handle in queue


def test_version_branch_tree_matches_reference_lists():
    # A small version branch tree:
    #
    #   v1 (a..f inserted)
    #   +-- v2 = v1 with c reprioritized 3 -> 1
    #   |     +-- v4 = v2 after pop_min
    #   |     +-- v5 = merge(v2, v3, 'left')
    #   +-- v3 = v1 with e reprioritized 3 -> 0
    #         +-- v6 = v3 with a reprioritized 5 -> 9
    v1, m1 = ppriorityqueue(), Reference()
    handles = {}
    for name, priority in [('a', 5), ('b', 3), ('c', 3), ('d', 7),
                           ('e', 3), ('f', 1)]:
        handles[name], v1 = v1.insert(name, priority)
        m1.insert(handles[name], name, priority)

    v2, m2 = v1.set_priority(handles['c'], 1), m1.clone()
    m2.set_priority(handles['c'], 1)

    v3, m3 = v1.set_priority(handles['e'], 0), m1.clone()
    m3.set_priority(handles['e'], 0)

    popped, v4 = v2.pop_min()
    m4 = m2.clone()
    assert (popped.handle, popped.payload, popped.priority) == m4.pop_min()

    # v2 and v3 modified *different* handles: each side's edit is newer than
    # the other's stale ancestor copy, so both edits survive the merge and
    # the divergences are reported, not treated as conflicts.
    v5, conflicts = v2.merge(v3, on_conflict='left')
    m5 = m2.merge(m3, 'left')
    assert conflicts == [
        MergeConflict(handles['c'], (1, 'c'), (3, 'c'), (1, 'c')),
        MergeConflict(handles['e'], (3, 'e'), (0, 'e'), (0, 'e')),
    ]
    assert v5.get(handles['c']).priority == 1
    assert v5.get(handles['e']).priority == 0

    v6, m6 = v3.set_priority(handles['a'], 9), m3.clone()
    m6.set_priority(handles['a'], 9)

    for version, model in [(v1, m1), (v2, m2), (v3, m3),
                           (v4, m4), (v5, m5), (v6, m6)]:
        assert_matches(version, model)


def test_merge_concurrent_conflict_error_reports_every_conflict():
    queue = ppriorityqueue()
    hx, queue = queue.insert('x', 1)
    hy, queue = queue.insert('y', 1)
    left = queue.set_priority(hx, 2).set_priority(hy, 2)
    right = queue.set_priority(hx, 3).set_priority(hy, 3)

    with pytest.raises(MergeConflictError) as excinfo:
        left.merge(right)
    assert excinfo.value.conflicts == [
        MergeConflict(hx, (2, 'x'), (3, 'x'), None),
        MergeConflict(hy, (2, 'y'), (3, 'y'), None),
    ]


def test_merge_concurrent_conflict_left_right_and_callable():
    queue = ppriorityqueue()
    hx, queue = queue.insert('x', 1)
    hy, queue = queue.insert('y', 4)
    left = queue.set_priority(hx, 2)
    right = queue.set_priority(hx, 3).set_priority(hy, 0)

    # hx diverges concurrently (same write epoch on both sides) -> policy;
    # hy was only touched by right, so right's fresher write wins and the
    # divergence is merely reported.
    merged, conflicts = left.merge(right, on_conflict='left')
    assert merged.get(hx).priority == 2
    assert merged.get(hy).priority == 0
    assert conflicts == [
        MergeConflict(hx, (2, 'x'), (3, 'x'), (2, 'x')),
        MergeConflict(hy, (4, 'y'), (0, 'y'), (0, 'y')),
    ]
    # The conflicting handle dequeues exactly once, not twice.
    assert len(merged) == 2
    assert [item.handle for item in merged].count(hx) == 1

    merged, conflicts = left.merge(right, on_conflict='right')
    assert merged.get(hx).priority == 3
    assert conflicts[0] == MergeConflict(hx, (2, 'x'), (3, 'x'), (3, 'x'))

    # Callable policy: keep whichever side has the lower priority.
    merged, conflicts = left.merge(
        right, on_conflict=lambda handle, lft, rgt: min(lft, rgt))
    assert merged.get(hx).priority == 2
    assert len(merged) == 2


def test_merge_rejects_bad_arguments():
    queue = ppriorityqueue()
    with pytest.raises(TypeError):
        queue.merge('not a queue')
    with pytest.raises(ValueError):
        queue.merge(queue, on_conflict='bogus')

    handle, queue = queue.insert('x', 1)
    left = queue.set_priority(handle, 2)
    right = queue.set_priority(handle, 3)
    with pytest.raises(TypeError):
        left.merge(right, on_conflict=lambda h, lft, rgt: 'not a pair')


def _tree_node_ids(root):
    ids = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        ids.add(id(node))
        stack.append(node.left)
        stack.append(node.right)
    return ids


def test_untouched_nodes_are_shared_between_versions():
    queue = ppriorityqueue()
    handles = []
    for i in range(10000):
        handle, queue = queue.insert(i, priority=i % 100)
        handles.append(handle)

    # One reprioritization and one pop must only pay O(log n) new nodes.
    updated = queue.set_priority(handles[5000], priority=-1)
    _, popped = updated.pop_min()

    for tree in ('_entries', '_order'):
        base_nodes = _tree_node_ids(getattr(queue, tree))
        updated_nodes = _tree_node_ids(getattr(updated, tree))
        popped_nodes = _tree_node_ids(getattr(popped, tree))
        assert len(base_nodes & updated_nodes) / len(updated_nodes) > 0.99
        assert len(updated_nodes & popped_nodes) / len(popped_nodes) > 0.99

    # The ancestor version is fully intact.
    assert len(queue) == 10000
    assert queue.get(handles[5000]).priority == 5000 % 100


def test_randomized_against_reference_model():
    rng = random.Random(20261005)
    versions = [(ppriorityqueue(), Reference())]

    for step in range(400):
        queue, model = versions[rng.randrange(len(versions))]
        op = rng.choice(['insert', 'insert', 'set_priority', 'pop_min',
                         'merge', 'check'])
        if op == 'insert' or not model.entries:
            payload = rng.randrange(1000)
            priority = rng.randrange(5)
            handle, new_queue = queue.insert(payload, priority)
            new_model = model.clone()
            new_model.insert(handle, payload, priority)
        elif op == 'set_priority':
            handle = rng.choice(list(model.entries))
            priority = rng.randrange(5)
            new_queue = queue.set_priority(handle, priority)
            new_model = model.clone()
            new_model.set_priority(handle, priority)
        elif op == 'pop_min':
            item, new_queue = queue.pop_min()
            new_model = model.clone()
            assert (item.handle, item.payload, item.priority) \
                == new_model.pop_min()
        elif op == 'merge':
            other_queue, other_model = versions[rng.randrange(len(versions))]
            policy = rng.choice(['left', 'right'])
            new_queue, _ = queue.merge(other_queue, on_conflict=policy)
            new_model = model.merge(other_model, policy)
        else:
            assert_matches(queue, model)
            continue
        versions.append((new_queue, new_model))
        assert_matches(new_queue, new_model)

    # Every surviving version in the pool must still match its model.
    for queue, model in versions:
        assert_matches(queue, model)

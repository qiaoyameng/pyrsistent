import pytest
from hypothesis import given, strategies as st

from pyrsistent import (
    PQHandle,
    PPriorityQueueConflictError,
    PPriorityQueueEmptyError,
    PPriorityQueueHandleError,
    PPriorityQueueMergeError,
    PriorityMergePolicy,
    pq,
    ppriorityqueue,
)


def drain(queue):
    result = []
    while queue:
        queue, item = queue.pop_min()
        result.append((item.priority, item.tie, item.value, item.handle))
    return result


def ref_insert(reference, value, priority, tie):
    reference.append((priority, tie, value))


def ref_pop(reference):
    ordered = sorted(reference)
    item = ordered[0]
    reference.remove(item)
    return item


def ref_set_priority(reference, handle_tie, priority):
    index = next(i for i, item in enumerate(reference) if item[1] == handle_tie)
    _, tie, value = reference[index]
    reference[index] = (priority, tie, value)


def test_stable_priority_order_and_popping():
    queue = ppriorityqueue()
    reference = []
    for tie, (value, priority) in enumerate(
            [('a', 1), ('b', 0), ('c', 0), ('d', 0)]):
        queue, _ = queue.insert(value, priority)
        ref_insert(reference, value, priority, tie)

    while queue:
        queue, item = queue.pop_min()
        priority, tie, value = ref_pop(reference)
        assert (item.priority, item.tie, item.value) == (priority, tie, value)

    assert not queue
    with pytest.raises(PPriorityQueueEmptyError):
        queue.pop_min()


def test_factory_and_none_value():
    queue = pq(('a', 1), (None, 0))
    reference = ppriorityqueue([('a', 1), (None, 0)])
    assert [(i.value, i.priority) for i in queue] == \
        [(i.value, i.priority) for i in reference]
    queue, item = queue.pop_min()
    assert item.value is None
    queue, item = queue.pop_min()
    assert item.value == 'a'


def test_empty_queue_operations_and_merge():
    empty = ppriorityqueue()
    queue, handle = empty.insert('a', 0)

    assert empty.merge(queue) is queue
    assert queue.merge(empty) is queue
    assert empty.merge(empty) is empty
    with pytest.raises(PPriorityQueueEmptyError):
        empty.peek_min()
    with pytest.raises(PPriorityQueueHandleError):
        empty.update_priority(handle, 1)


def test_version_branch_tree_matches_reference_list():
    queue = ppriorityqueue()
    reference = []
    handles = []
    for tie, item in enumerate([('a', 5), ('b', 1), ('c', 3)]):
        queue, handle = queue.insert(*item)
        handles.append(handle)
        ref_insert(reference, item[0], item[1], tie)

    low = queue.update_priority(handles[0], 0)
    high = queue.update_priority(handles[0], 9)
    pop_b, popped = queue.pop_min()
    assert popped.handle == handles[1]

    low_ref = reference.copy()
    high_ref = reference.copy()
    pop_b_ref = reference.copy()
    ref_set_priority(low_ref, handles[0].item_id, 0)
    ref_set_priority(high_ref, handles[0].item_id, 9)
    priority, tie, value = ref_pop(pop_b_ref)
    assert (priority, tie, value) == (1, 1, 'b')

    assert [(p, t, v) for p, t, v, _ in drain(low)] == sorted(low_ref)
    assert [(p, t, v) for p, t, v, _ in drain(high)] == sorted(high_ref)
    assert [(p, t, v) for p, t, v, _ in drain(pop_b)] == sorted(pop_b_ref)
    assert [(p, t, v) for p, t, v, _ in drain(queue)] == sorted(reference)

    with pytest.raises(PPriorityQueueConflictError) as exc:
        low.merge(high)
    assert exc.value.conflicts[0].handle == handles[0]

    merged = low.merge(high, PriorityMergePolicy.MIN_PRIORITY)
    assert [(p, t, v) for p, t, v, _ in drain(merged)] == sorted(low_ref)

    final = merged.merge(pop_b, PriorityMergePolicy.LEFT)
    final_ref = low_ref.copy()
    final_ref.remove((1, 1, 'b'))
    assert [(p, t, v) for p, t, v, _ in drain(final)] == sorted(final_ref)
    assert len(final) == 2


def test_branches_can_change_same_handle_independently():
    queue, handle = ppriorityqueue().insert('a', 5)
    left = queue.update_priority(handle, 1)
    right = queue.update_priority(handle, 10)

    assert queue.peek_min().priority == 5
    assert left.peek_min().priority == 1
    assert right.peek_min().priority == 10

    merged_left = left.merge(right, PriorityMergePolicy.LEFT)
    merged_right = left.merge(right, PriorityMergePolicy.RIGHT)
    assert merged_left.peek_min().priority == 1
    assert merged_right.peek_min().priority == 10


def test_merged_branch_keeps_future_remerge_conflicts_explicit():
    queue, handle = ppriorityqueue().insert('a', 1)
    left = queue.update_priority(handle, 0)
    right = queue.update_priority(handle, 2)
    merged = left.merge(right, PriorityMergePolicy.LEFT)

    assert merged.merge(right, PriorityMergePolicy.LEFT).peek_min().priority == 0
    assert merged.merge(right, PriorityMergePolicy.RIGHT).peek_min().priority == 2
    with pytest.raises(PPriorityQueueConflictError):
        merged.merge(right)


def test_merge_dedupes_shared_items_and_accepts_fast_forward():
    queue, _ = ppriorityqueue().insert('a', 1)
    queue, handle_b = queue.insert('b', 0)
    left = queue.set_value(handle_b, 'left')
    right = queue.insert('c', 2)[0]

    merged = left.merge(right, PriorityMergePolicy.LEFT)
    assert len(merged) == 3
    assert len({item.handle for item in merged}) == 3

    assert queue.merge(left) is left
    assert left.merge(queue) is left


def test_merge_reports_deletion_vs_update_explicitly():
    queue, handle = ppriorityqueue().insert('a', 1)
    updated = queue.update_priority(handle, 0)
    popped, _ = queue.pop_min()

    assert updated.merge(popped, PriorityMergePolicy.LEFT).peek_min().priority == 0
    assert not updated.merge(popped, PriorityMergePolicy.RIGHT)
    with pytest.raises(PPriorityQueueConflictError):
        updated.merge(popped, PriorityMergePolicy.MIN_PRIORITY)


def test_handles_are_rejected_explicitly():
    queue, handle = ppriorityqueue().insert('a', 1)
    other, other_handle = ppriorityqueue().insert('b', 1)
    popped, _ = queue.pop_min()

    for bad_handle in (None, 1, object()):
        with pytest.raises(PPriorityQueueHandleError):
            queue.update_priority(bad_handle, 0)

    with pytest.raises(PPriorityQueueHandleError):
        queue.update_priority(other_handle, 0)
    with pytest.raises(PPriorityQueueHandleError):
        popped.update_priority(handle, 0)
    with pytest.raises(PPriorityQueueMergeError):
        queue.merge(other)
    assert isinstance(handle, PQHandle)


def _treap_nodes(node):
    if node is None:
        return set()
    return {id(node)} | _treap_nodes(node.left) | _treap_nodes(node.right)


def test_most_unchanged_treap_nodes_are_shared():
    queue = ppriorityqueue((value, value) for value in range(500))
    handle = PQHandle(queue._family.family_id, item_id=400)
    updated = queue.update_priority(handle, -1)

    original_nodes = _treap_nodes(queue._heap)
    updated_nodes = _treap_nodes(updated._heap)
    shared = original_nodes & updated_nodes

    assert len(shared) / len(original_nodes) > 0.85


def test_merge_shares_most_unchanged_branch_nodes():
    queue = ppriorityqueue((value, value) for value in range(500))
    left_handle = PQHandle(queue._family.family_id, item_id=100)
    right_handle = PQHandle(queue._family.family_id, item_id=400)
    left = queue.update_priority(left_handle, -10)
    right = queue.update_priority(right_handle, 510)
    merged = left.merge(right, PriorityMergePolicy.LEFT)

    branch_nodes = _treap_nodes(left._heap) | _treap_nodes(right._heap)
    merged_nodes = _treap_nodes(merged._heap)
    shared = branch_nodes & merged_nodes

    assert len(shared) / len(merged_nodes) > 0.85


@given(data=st.data())
def test_random_branches_and_merges_never_duplicate_handles(data):
    queue = ppriorityqueue()
    for _ in range(data.draw(st.integers(min_value=2, max_value=10))):
        priority = data.draw(st.integers(min_value=0, max_value=4))
        queue, _ = queue.insert(data.draw(st.text(max_size=4)), priority)

    branches = [queue]
    for _ in range(data.draw(st.integers(min_value=2, max_value=8))):
        base = data.draw(st.sampled_from(branches))
        active = list(base)
        if active and data.draw(st.booleans()):
            item = data.draw(st.sampled_from(active))
            updated = base.update_priority(
                item.handle, data.draw(st.integers(min_value=0, max_value=4)))
            branches.append(updated)
            if data.draw(st.booleans()):
                branches.append(base.pop_min()[0])
        else:
            inserted, _ = base.insert(
                data.draw(st.text(max_size=4)),
                data.draw(st.integers(min_value=0, max_value=4)))
            branches.append(inserted)

    left, right = data.draw(st.sampled_from(branches)), data.draw(
        st.sampled_from(branches))
    policy = data.draw(st.sampled_from([
        PriorityMergePolicy.LEFT,
        PriorityMergePolicy.RIGHT,
        PriorityMergePolicy.MIN_PRIORITY,
    ]))
    try:
        merged = left.merge(right, policy)
    except PPriorityQueueConflictError:
        return

    ordered = [
        (item.priority, item.tie, item.handle)
        for item in merged
    ]
    assert ordered == sorted(ordered)
    assert len({item.handle for item in merged}) == len(merged)
    assert len(merged._active) == len(merged)

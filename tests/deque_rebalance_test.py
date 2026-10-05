"""
Tests for the smooth (incremental) rebalancing of PDeque.

These tests do not rely on average timings. They count the basic internal
steps (stream cell creations and lazy cell forcings) performed by every
single operation and assert a hard, size independent upper bound. They also
verify that delayed rebalancing work is shared between branches forked from
the same version, and that all versions keep their contents intact.
"""
import random

import pytest

from pyrsistent import pdeque, dq
from pyrsistent._pdeque import step_counter


# Maximum number of basic internal steps a single endpoint operation may
# perform, regardless of the deque size. The measured worst case is 13; this
# leaves headroom while staying far below any tested deque size, proving that
# no endpoint operation ever scans the whole queue.
MAX_STEPS_PER_OP = 32

# Appending to a full maxlen bounded deque performs two sub-operations (a pop
# and a push), so it gets a doubled, still constant, budget.
MAX_STEPS_PER_TRUNCATING_OP = 64


def count_steps(fn):
    step_counter.reset()
    step_counter.enabled = True
    try:
        result = fn()
    finally:
        step_counter.enabled = False
    return result, step_counter.count


@pytest.mark.parametrize('size', [100, 1000, 10000])
def test_single_operations_do_bounded_work(size):
    # Drain a deque that is heavy on one side from the other end. This is the
    # pattern that used to trigger an O(n) reversal in a single call.
    deque = pdeque(range(size))
    while len(deque) > 0:
        deque, steps = count_steps(deque.popleft)
        assert steps <= MAX_STEPS_PER_OP
    assert len(deque) == 0

    # Drain from alternating ends.
    deque = pdeque(range(size))
    while len(deque) > 0:
        deque, steps = count_steps(deque.popleft)
        assert steps <= MAX_STEPS_PER_OP
        if len(deque) > 0:
            deque, steps = count_steps(deque.pop)
            assert steps <= MAX_STEPS_PER_OP
    assert len(deque) == 0

    # Sawtooth: interleave appends and pops on both ends.
    deque = pdeque(range(size)).popleft(size // 2)
    for i in range(size):
        deque, steps = count_steps(lambda: deque.appendleft(i))
        assert steps <= MAX_STEPS_PER_OP
        deque, steps = count_steps(lambda: deque.append(i))
        assert steps <= MAX_STEPS_PER_OP
        deque, steps = count_steps(deque.popleft)
        assert steps <= MAX_STEPS_PER_OP
        deque, steps = count_steps(deque.pop)
        assert steps <= MAX_STEPS_PER_OP


@pytest.mark.parametrize('maxlen', [0, 1, 5, 1000])
def test_maxlen_operations_do_bounded_work(maxlen):
    deque = pdeque(range(maxlen), maxlen=maxlen)
    for i in range(4 * maxlen + 10):
        deque, steps = count_steps(lambda: deque.append(i))
        assert steps <= MAX_STEPS_PER_TRUNCATING_OP
        assert len(deque) <= maxlen

        deque, steps = count_steps(lambda: deque.appendleft(i))
        assert steps <= MAX_STEPS_PER_TRUNCATING_OP
        assert len(deque) <= maxlen


def test_interleaved_operations_match_model_and_stay_bounded():
    rng = random.Random(20261005)
    for maxlen in (None, 0, 1, 3, 17):
        deque = pdeque(maxlen=maxlen)
        model = []
        versions = [(deque, list(model))]

        for i in range(2000):
            op = rng.randrange(8)
            value = rng.randrange(50)
            full = maxlen is not None and len(model) == maxlen
            budget = MAX_STEPS_PER_TRUNCATING_OP if full else MAX_STEPS_PER_OP

            if op == 0:
                deque, steps = count_steps(lambda: deque.append(value))
                if not full:
                    model.append(value)
                elif maxlen:
                    model.pop(0)
                    model.append(value)
            elif op == 1:
                deque, steps = count_steps(lambda: deque.appendleft(value))
                if not full:
                    model.insert(0, value)
                elif maxlen:
                    model.pop()
                    model.insert(0, value)
            elif op == 2:
                deque, steps = count_steps(deque.pop)
                if model:
                    model.pop()
            elif op == 3:
                deque, steps = count_steps(deque.popleft)
                if model:
                    model.pop(0)
            elif op == 4:
                k = rng.randrange(4)
                deque, steps = count_steps(lambda: deque.pop(k))
                del model[len(model) - min(k, len(model)):]
                budget = MAX_STEPS_PER_OP * max(k, 1)
            elif op == 5:
                k = rng.randrange(4)
                deque, steps = count_steps(lambda: deque.popleft(k))
                del model[:k]
                budget = MAX_STEPS_PER_OP * max(k, 1)
            elif op == 6:
                values = [rng.randrange(50) for _ in range(rng.randrange(5))]
                deque, steps = count_steps(lambda: deque.extend(values))
                model.extend(values)
                if maxlen is not None:
                    del model[:max(0, len(model) - maxlen)]
                budget = MAX_STEPS_PER_TRUNCATING_OP * max(len(values), 1)
            else:
                values = [rng.randrange(50) for _ in range(rng.randrange(5))]
                deque, steps = count_steps(lambda: deque.extendleft(values))
                model[:0] = reversed(values)
                if maxlen is not None:
                    del model[maxlen:]
                budget = MAX_STEPS_PER_TRUNCATING_OP * max(len(values), 1)

            assert steps <= budget, 'operation {0} at size {1} took {2} steps'.format(op, len(model), steps)
            assert list(deque) == model
            assert deque.maxlen == maxlen

            if i % 200 == 0:
                versions.append((deque, list(model)))

        # All versions created along the way must remain intact.
        for version, expected in versions:
            assert list(version) == expected


def test_delayed_rebalancing_work_is_shared_between_branches():
    # The last popleft triggers a rebalance, leaving parent with pending
    # (not yet realized) rebalancing work.
    parent = pdeque(range(2000)).popleft(501)

    branch_a, steps_a = count_steps(lambda: parent.popleft(200))
    branch_b, steps_b = count_steps(lambda: parent.popleft(200))

    expected = list(parent)[200:]
    assert list(branch_a) == expected
    assert list(branch_b) == expected

    # The first branch pays for realizing the shared delayed work. The second
    # branch, forked from the same version, reuses the memoized result instead
    # of re-reversing the old lists.
    assert steps_a > 500
    assert steps_b * 4 < steps_a


def test_forking_from_old_versions_keeps_all_contents_and_bounds_work():
    rng = random.Random(777)
    versions = [(pdeque(range(200)), list(range(200)))]

    for _ in range(30):
        parent, parent_model = versions[rng.randrange(len(versions))]
        branch = parent
        model = list(parent_model)

        for _ in range(60):
            op = rng.randrange(4)
            value = rng.randrange(1000)
            if op == 0:
                branch, steps = count_steps(lambda: branch.append(value))
                model.append(value)
            elif op == 1:
                branch, steps = count_steps(lambda: branch.appendleft(value))
                model.insert(0, value)
            elif op == 2:
                branch, steps = count_steps(branch.pop)
                if model:
                    model.pop()
            else:
                branch, steps = count_steps(branch.popleft)
                if model:
                    model.pop(0)

            assert steps <= MAX_STEPS_PER_OP
            assert list(branch) == model

        versions.append((branch, list(model)))

    for version, expected in versions:
        assert list(version) == expected


def test_iteration_and_ends_after_rebalancing():
    # Iteration may do linear work, but it must visit elements in order and
    # the endpoints must stay correct while rebalances are in flight.
    deque = dq(*range(100))
    for _ in range(90):
        deque = deque.popleft()
    assert list(deque) == list(range(90, 100))
    assert deque.left == 90
    assert deque.right == 99
    assert list(deque.reverse()) == list(range(99, 89, -1))

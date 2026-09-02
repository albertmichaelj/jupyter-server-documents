/* -----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

import {
  bumpConnectionEpoch,
  getConnectionEpoch,
  shouldBumpEpoch,
  ExecutionChain,
  EPOCH_BUMP_MIN_DOWNTIME_MS
} from '../docprovider/executionChain';

// The epoch registry is module-level state shared across tests, so every test
// uses its own room names.

describe('connection epochs', () => {
  it('starts at 0 for a room that has never connected', () => {
    expect(getConnectionEpoch('epoch-room-unseen')).toBe(0);
  });

  it('increments on every bump', () => {
    const room = 'epoch-room-bumps';
    bumpConnectionEpoch(room);
    expect(getConnectionEpoch(room)).toBe(1);
    bumpConnectionEpoch(room);
    expect(getConnectionEpoch(room)).toBe(2);
  });

  it('is independent per room', () => {
    bumpConnectionEpoch('epoch-room-a');
    expect(getConnectionEpoch('epoch-room-b')).toBe(0);
  });
});

describe('shouldBumpEpoch', () => {
  it('bumps on a first connection (no recorded disconnect)', () => {
    expect(shouldBumpEpoch(null, 1000)).toBe(true);
  });

  it('does not bump after a short blip — the room is guaranteed alive, and the chain protects a request in flight across the blip', () => {
    expect(shouldBumpEpoch(1000, 1000 + 2000)).toBe(false);
    expect(shouldBumpEpoch(1000, 1000 + EPOCH_BUMP_MIN_DOWNTIME_MS - 1)).toBe(
      false
    );
  });

  it('bumps after downtime long enough that the room may have been freed', () => {
    expect(shouldBumpEpoch(1000, 1000 + EPOCH_BUMP_MIN_DOWNTIME_MS)).toBe(true);
  });
});

describe('ExecutionChain', () => {
  it('chains successive requests within one epoch', () => {
    const chain = new ExecutionChain();
    expect(chain.next('doc:client', 1, 'r1')).toBeUndefined();
    expect(chain.next('doc:client', 1, 'r2')).toBe('r1');
    expect(chain.next('doc:client', 1, 'r3')).toBe('r2');
  });

  it('breaks the chain exactly once on an epoch change', () => {
    const chain = new ExecutionChain();
    chain.next('doc:client', 1, 'r1');
    // Reconnect happened: the first request afterwards must NOT name r1 —
    // the recreated room has never seen it and would wait out the
    // predecessor timeout, then 408.
    expect(chain.next('doc:client', 2, 'r2')).toBeUndefined();
    // Subsequent requests chain normally among themselves, preserving
    // ordering within the new connection.
    expect(chain.next('doc:client', 2, 'r3')).toBe('r2');
  });

  it('clear() breaks the chain for the next request only', () => {
    const chain = new ExecutionChain();
    chain.next('doc:client', 1, 'r1');
    chain.clear('doc:client', 1);
    expect(chain.next('doc:client', 1, 'r2')).toBeUndefined();
    expect(chain.next('doc:client', 1, 'r3')).toBe('r2');
  });

  // Deliberate reversal of an earlier assertion. It used to require that a
  // late failure leave a newer in-flight registration alone, on the theory
  // that the newer request might still be healthy. Within one epoch it never
  // is: the chain is a linked list, so r2 named r1 as its predecessor, and a
  // server that never enqueued r1 will make r2 wait for it until the
  // predecessor timeout and fail. Keeping r2 registered hands that poison to
  // r3, and so on for as long as the user keeps retrying.
  it('a failure unchains the doomed successor it poisoned', () => {
    const chain = new ExecutionChain();
    chain.next('doc:client', 1, 'r1');
    expect(chain.next('doc:client', 1, 'r2')).toBe('r1');
    chain.clear('doc:client', 1);
    expect(chain.next('doc:client', 1, 'r3')).toBeUndefined();
  });

  // The property the id-scoping was actually reaching for, kept intact: a
  // reconnect starts a fresh chain, and a late failure from before it must
  // not unchain the new one.
  it('a failure from an older epoch leaves a newer chain alone', () => {
    const chain = new ExecutionChain();
    chain.next('doc:client', 1, 'r1');
    expect(chain.next('doc:client', 2, 'r2')).toBeUndefined();
    chain.clear('doc:client', 1);
    expect(chain.next('doc:client', 2, 'r3')).toBe('r2');
  });

  // The production incident, in miniature: Run All issues every request
  // before any response returns, so the first failure arrives when `_last` is
  // already the last of the batch. Under id-scoping every clear was a no-op
  // and the chain stayed poisoned indefinitely.
  it('recovers after a Run All burst in which every request fails', () => {
    const chain = new ExecutionChain();
    const ids = ['r1', 'r2', 'r3', 'r4', 'r5'];
    ids.forEach(id => chain.next('doc:client', 1, id));
    // Responses come back in order; the first one clears the whole chain.
    chain.clear('doc:client', 1);
    expect(chain.next('doc:client', 1, 'next-run')).toBeUndefined();
    // Later failures from the same dead batch are harmless no-ops for the
    // fresh chain that has since been started.
    chain.clear('doc:client', 1);
    expect(chain.next('doc:client', 1, 'after')).toBeUndefined();
  });

  it('keeps chains independent per docKey (other users, other docs)', () => {
    const chain = new ExecutionChain();
    chain.next('doc:alice', 1, 'a1');
    // Bob's first request is unaffected by Alice's chain…
    expect(chain.next('doc:bob', 1, 'b1')).toBeUndefined();
    // …and an epoch change seen by Alice does not disturb Bob.
    expect(chain.next('doc:alice', 2, 'a2')).toBeUndefined();
    expect(chain.next('doc:bob', 1, 'b2')).toBe('b1');
  });
});

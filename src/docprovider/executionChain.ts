/* -----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Execution-request chaining that does not survive a reconnect.
 *
 * The server-side execution API orders requests with `previous_request_id`,
 * but the enqueued-request history it checks against lives in the YRoom. If
 * the room is garbage-collected and later recreated (e.g. while a laptop lid
 * is closed), that history is empty — a request chained to a pre-recreation
 * id waits the full predecessor timeout and fails with 408, even though the
 * predecessor can never arrive.
 *
 * The client is the one party that knows when its connection was
 * re-established, so the chain is keyed by a per-room "connection epoch"
 * bumped on every (re)connect: the first request after a reconnect goes out
 * unchained, and subsequent requests chain normally among themselves.
 *
 * The epoch registry is module-local — per browser tab, NOT shared-model
 * state — so one user's reconnect cannot break another user's chain. Ordering
 * across users is not affected either way: each client only ever names its
 * own previous request id.
 */

const connectionEpochs = new Map<string, number>();

/**
 * Minimum WebSocket downtime before a reconnect invalidates chains.
 *
 * A room is only freed after `YRoom.inactivity_timeout` (default 60s) with no
 * connected clients, and its inactivity clock cannot start before this client
 * disconnected — so a reconnect after less downtime than that is guaranteed
 * to find the room, and its request history, alive. Breaking the chain on
 * such a blip would give up FIFO protection for a request in flight across
 * it, for no benefit. 45s leaves margin for client-side measurement error.
 *
 * A server restart inside the window still loses the history; that costs one
 * recoverable 408 (the chain clears on failure), same as before this module
 * existed.
 */
export const EPOCH_BUMP_MIN_DOWNTIME_MS = 45_000;

/**
 * Whether a transition to 'connected' should invalidate chains: yes on the
 * first connection or when downtime cannot be measured (both conservative —
 * there is nothing in flight to protect on a first connection), and on any
 * reconnect after downtime long enough that the room may have been freed.
 */
export function shouldBumpEpoch(
  disconnectedAt: number | null,
  now: number
): boolean {
  return (
    disconnectedAt === null ||
    now - disconnectedAt >= EPOCH_BUMP_MIN_DOWNTIME_MS
  );
}

/**
 * Record a (re)connect for a room. Called by the WebSocket provider on a
 * transition to 'connected' that passes the `shouldBumpEpoch` gate.
 */
export function bumpConnectionEpoch(roomName: string): void {
  connectionEpochs.set(roomName, getConnectionEpoch(roomName) + 1);
}

/**
 * The current connection epoch for a room; 0 if it has never connected.
 */
export function getConnectionEpoch(roomName: string): number {
  return connectionEpochs.get(roomName) ?? 0;
}

/**
 * Tracks the last request id per document+client, invalidated by epoch.
 */
export class ExecutionChain {
  /**
   * Record `requestId` as the newest request for `docKey` and return the id
   * it should chain to: the previous request iff it was issued in the same
   * connection epoch, else undefined (start a fresh chain).
   */
  next(docKey: string, epoch: number, requestId: string): string | undefined {
    const prev = this._last.get(docKey);
    this._last.set(docKey, { epoch, requestId });
    return prev && prev.epoch === epoch ? prev.requestId : undefined;
  }

  /**
   * Forget the chain for `docKey` iff its newest entry is `requestId`.
   * Called when a request fails: an id that was never enqueued on the server
   * must not be named as a predecessor. Scoped to the failed id because a
   * newer request may already be registered (and in flight) — a stale
   * failure must not unchain it.
   */
  clear(docKey: string, requestId: string): void {
    if (this._last.get(docKey)?.requestId === requestId) {
      this._last.delete(docKey);
    }
  }

  private _last = new Map<string, { epoch: number; requestId: string }>();
}

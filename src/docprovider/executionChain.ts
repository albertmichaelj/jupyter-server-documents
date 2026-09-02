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
 * Minimum time since the client was last KNOWN ALIVE before a reconnect
 * invalidates chains.
 *
 * Downtime is measured from a liveness stamp refreshed every
 * `ALIVE_STAMP_INTERVAL_MS` while connected — NOT from the 'disconnected'
 * event. That distinction is the whole point: JS is frozen during laptop
 * sleep (and Chrome tab-freeze), so the disconnect for a connection that
 * died hours ago is only *delivered* at wake, seconds before the reconnect —
 * measured that way, a 20-minute lid close looks like a 3-second blip. The
 * interval stamp freezes with the page instead, so at wake the gap it shows
 * IS the sleep.
 *
 * Why gate at all: a room is only freed after `YRoom.inactivity_timeout`
 * (default 60s) plus a sweep, so a short awake blip almost always finds the
 * room, and its request history, alive — and there the chain must survive,
 * because it is the FIFO protection for a request in flight across the blip.
 * 45s leaves margin under the 60s bound for stamp granularity.
 *
 * Best-effort, not airtight: a client that is connected but idle can sit in
 * a room that is already inactive (room activity is edit-based, not
 * presence-based), so a sub-45s blip that straddles a sweep tick can still
 * find the room freed; a server restart inside the window loses the history
 * too. Either costs one recoverable 408 — the chain clears on failure — the
 * same self-heal that existed before this module.
 */
export const EPOCH_BUMP_MIN_DOWNTIME_MS = 45_000;

/**
 * How often the provider refreshes its liveness stamp while connected. Must
 * be well under `EPOCH_BUMP_MIN_DOWNTIME_MS`: an awake blip measures at most
 * its real length plus one interval.
 */
export const ALIVE_STAMP_INTERVAL_MS = 10_000;

/**
 * Whether a transition to 'connected' should invalidate chains: yes on the
 * first connection or when liveness was never stamped (both conservative —
 * there is nothing in flight to protect on a first connection), and on any
 * reconnect after long enough that the room may have been freed.
 */
export function shouldBumpEpoch(
  lastAliveAt: number | null,
  now: number
): boolean {
  return (
    lastAliveAt === null || now - lastAliveAt >= EPOCH_BUMP_MIN_DOWNTIME_MS
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
   * Forget the chain for `docKey` iff its newest entry is from `epoch`.
   * Called when a request fails: an id that was never enqueued on the server
   * must not be named as a predecessor.
   *
   * Scoped to the EPOCH, not to the failed request id. Scoping to the id was
   * too narrow and cost a user a dead notebook: within one epoch the chain is
   * a linked list, so a newer entry chained (transitively) onto the failed
   * request, and the server makes a successor wait for a predecessor that was
   * never enqueued. That successor is therefore already doomed — there is no
   * healthy in-flight request to protect, and leaving it registered hands the
   * poison to everything issued after it.
   *
   * Run All makes this permanent rather than transient. It issues N requests
   * before any response returns, so `_last` is already the Nth by the time the
   * first failure arrives and every id-scoped clear is a no-op; each retry
   * inside the server's 10s predecessor timeout then re-chains onto a doomed
   * request. Observed in production: every execute returning 408 "Timed out
   * waiting for previous_request_id to be enqueued" for eleven minutes, with
   * no cell ever running. Nothing server-side could clear it — the poisoned
   * state is in this map — so closing the notebook, shutting down the kernel
   * and restarting it all failed. Only reloading the browser page recovered.
   *
   * The epoch is what the original scoping was really reaching for: a newer
   * request from a LATER epoch began a fresh chain after a reconnect and must
   * survive a late failure from the old one.
   */
  clear(docKey: string, epoch: number): void {
    if (this._last.get(docKey)?.epoch === epoch) {
      this._last.delete(docKey);
    }
  }

  private _last = new Map<string, { epoch: number; requestId: string }>();
}

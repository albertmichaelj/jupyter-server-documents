/**
 * Regression tests for the divergent-history repair in `applyServerUpdate`.
 *
 * The repair must be IDEMPOTENT: repeating it against the same server state
 * must never delete content the server owns. The pre-fix implementation
 * cleared the full ordered range, so a second pass (reached whenever the
 * first repair's SS2 reply is lost) deleted the server's own items and, once
 * synced, emptied the document on disk. Reproduced in production as notebooks
 * truncated to a single blank cell.
 */
import * as Y from 'yjs';
import {
  applyServerUpdate,
  hasDivergentHistory
} from '../docprovider/yprovider';

/** Build a "server" doc with `text` in a top-level Y.Text named `source`. */
function makeServerDoc(text: string): Y.Doc {
  const doc = new Y.Doc();
  doc.getText('source').insert(0, text);
  return doc;
}

/** Build a divergent "client" doc: same visible text, its own history. */
function makeDivergentClientDoc(text: string): Y.Doc {
  const doc = new Y.Doc();
  doc.getText('source').insert(0, text);
  return doc;
}

describe('applyServerUpdate divergent repair', () => {
  const TEXT = 'the quick brown fox';

  it('repairs a divergent client to exactly the server content', () => {
    const server = makeServerDoc(TEXT);
    const client = makeDivergentClientDoc(TEXT);
    const serverUpdate = Y.encodeStateAsUpdate(server);
    const serverSV = Y.encodeStateVector(server);

    expect(hasDivergentHistory(client, serverSV)).toBe(true);
    applyServerUpdate(client, serverUpdate, true, undefined, serverSV);

    expect(client.getText('source').toString()).toBe(TEXT);
  });

  it('is IDEMPOTENT: a second repair pass must not delete server content', () => {
    const server = makeServerDoc(TEXT);
    const client = makeDivergentClientDoc(TEXT);
    const serverUpdate = Y.encodeStateAsUpdate(server);
    const serverSV = Y.encodeStateVector(server);

    // Pass 1: normal repair.
    applyServerUpdate(client, serverUpdate, true, undefined, serverSV);
    expect(client.getText('source').toString()).toBe(TEXT);

    // The client's tombstones were "lost in transit" (server never applied
    // the SS2 reply), so the next handshake sees the client divergent AGAIN
    // and runs the repair against the same server state.
    expect(hasDivergentHistory(client, serverSV)).toBe(true);
    applyServerUpdate(client, serverUpdate, true, undefined, serverSV);

    // Pre-fix, the full-range clear deleted the server's items here and the
    // near-empty diff could not resurrect them: content became ''.
    expect(client.getText('source').toString()).toBe(TEXT);

    // And a third pass, for good measure.
    applyServerUpdate(client, serverUpdate, true, undefined, serverSV);
    expect(client.getText('source').toString()).toBe(TEXT);
  });

  it('preserves the server-known prefix of a partially-covered client', () => {
    // Client synced its first edit to the server, then typed more offline.
    const client = new Y.Doc();
    client.getText('source').insert(0, 'synced.');
    const server = new Y.Doc();
    Y.applyUpdate(server, Y.encodeStateAsUpdate(client));
    const serverSV = Y.encodeStateVector(server); // covers 'synced.' only
    client.getText('source').insert(7, ' offline-tail');

    const serverUpdate = Y.encodeStateAsUpdate(
      server,
      Y.encodeStateVector(client)
    );
    applyServerUpdate(client, serverUpdate, true, undefined, serverSV);

    // The covered prefix survives; the uncovered offline tail is sacrificed
    // (persisted file is the source of truth), matching the repair contract.
    expect(client.getText('source').toString()).toBe('synced.');
  });

  it('non-divergent path is a plain applyUpdate that preserves local edits', () => {
    const client = new Y.Doc();
    client.getText('source').insert(0, 'base');
    const server = new Y.Doc();
    Y.applyUpdate(server, Y.encodeStateAsUpdate(client));
    client.getText('source').insert(4, ' + local edit');

    const serverUpdate = Y.encodeStateAsUpdate(
      server,
      Y.encodeStateVector(client)
    );
    applyServerUpdate(client, serverUpdate, false, undefined);

    expect(client.getText('source').toString()).toBe('base + local edit');
  });

  it('leaves Y.Map content untouched during repair', () => {
    const server = makeServerDoc(TEXT);
    server.getMap('meta').set('kernelspec', 'python3');
    const client = makeDivergentClientDoc(TEXT);
    client.getMap('meta').set('kernelspec', 'python3');

    const serverUpdate = Y.encodeStateAsUpdate(server);
    const serverSV = Y.encodeStateVector(server);
    applyServerUpdate(client, serverUpdate, true, undefined, serverSV);
    applyServerUpdate(client, serverUpdate, true, undefined, serverSV);

    expect(client.getMap('meta').get('kernelspec')).toBe('python3');
  });
});

describe('hasDivergentHistory clock comparison', () => {
  it('flags a non-self clientID whose clock exceeds server coverage', () => {
    // Two tabs hold unequal amounts of a dead session's history. The
    // less-complete tab repairs first and teaches the recreated room a
    // PREFIX of the stale clientID. When the fuller tab then reconnects,
    // presence-only detection sees every clientID covered and skips the
    // repair — its stale tail syncs as live items next to the server's
    // re-authored copy: permanent duplication on disk.
    const staleSession = new Y.Doc();
    staleSession.getText('source').insert(0, 'one|');
    const prefix = Y.encodeStateAsUpdate(staleSession);
    staleSession.getText('source').insert(4, 'two|');
    const full = Y.encodeStateAsUpdate(staleSession);

    const fullerTab = new Y.Doc();
    Y.applyUpdate(fullerTab, full);

    const serverDoc = new Y.Doc();
    serverDoc.getText('source').insert(0, 'one|two|'); // re-authored from disk
    Y.applyUpdate(serverDoc, prefix); // learned via the first repairer
    const serverSV = Y.encodeStateVector(serverDoc);

    expect(hasDivergentHistory(fullerTab, serverSV)).toBe(true);
  });

  it("does not flag the doc's own offline-edit overhang", () => {
    const client = new Y.Doc();
    client.getText('source').insert(0, 'synced.');
    const server = new Y.Doc();
    Y.applyUpdate(server, Y.encodeStateAsUpdate(client));
    const serverSV = Y.encodeStateVector(server);

    client.getText('source').insert(7, ' offline'); // legitimate offline work
    expect(hasDivergentHistory(client, serverSV)).toBe(false);
  });
});

describe('repair spares post-repair own insertions', () => {
  it('a second pass must not delete content created after the first', () => {
    const TEXT = 'the quick brown fox';
    const server = makeServerDoc(TEXT);
    const client = makeDivergentClientDoc(TEXT);
    const serverUpdate = Y.encodeStateAsUpdate(server);
    const serverSV = Y.encodeStateVector(server);

    // Pass 1 repairs; its SS2 reply is then lost in transit.
    applyServerUpdate(client, serverUpdate, true, undefined, serverSV);
    expect(client.getText('source').toString()).toBe(TEXT);

    // The user keeps working — this content postdates the dead room and
    // cannot duplicate anything the server re-authored.
    client.getText('source').insert(TEXT.length, ' NEW-WORK');

    // Pass 2 runs against the same server state (still divergent: the
    // server never learned this client's IDs).
    expect(hasDivergentHistory(client, serverSV)).toBe(true);
    applyServerUpdate(client, serverUpdate, true, undefined, serverSV);

    expect(client.getText('source').toString()).toBe(TEXT + ' NEW-WORK');
  });
});

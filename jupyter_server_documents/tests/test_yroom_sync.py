"""
Integration tests for YRoom sync handshake behavior.

These tests use a FakeWebSocket client that simulates the client side of the
Yjs sync protocol against a real YRoom instance. They verify:

1. Normal sync handshake completes successfully.
2. Divergent client detection works correctly.
3. Divergent client handshake resolves content duplication.
4. Timeout fires if client never sends SS2.
5. Update buffer pauses/resumes correctly during divergent handshake.
6. No data loss when mutations occur during the sync handshake.
"""

from __future__ import annotations

import asyncio
import pycrdt
from pycrdt import Doc, Text
from pycrdt import YMessageType, YSyncMessageType as YSyncMessageSubtype
import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...conftest import MakeYRoom
    from jupyter_server_documents.rooms.yroom import YRoom


class FakeWebSocket:
    """A fake WebSocket that records messages sent by the server and can
    replay the client side of the Yjs sync handshake.

    Usage:
        ws = FakeWebSocket()
        # optionally pre-populate with divergent content:
        ws.doc["source"] += "hello world"

        client_id = yroom.clients.add(ws)
        # server processes SS1 via the message queue or handle_sync()
        # then inspect ws.messages for what the server sent
    """

    def __init__(self, doc: Doc | None = None):
        self.doc = doc or Doc()
        if "source" not in self.doc:
            self.doc["source"] = Text()
        self.messages: list[bytes] = []
        self.closed = False
        self.close_code: int | None = None
        # Required by YjsClientGroup.get() check
        self.ws_connection = True

    def write_message(self, message: bytes, binary: bool = True) -> None:
        """Called by the server to send a message to this client."""
        self.messages.append(message)

    def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed = True
        self.close_code = code

    def build_ss1(self) -> bytes:
        """Build an SS1 message from this client's YDoc."""
        return pycrdt.create_sync_message(self.doc)

    def process_server_messages(self) -> bytes | None:
        """Process all messages from the server (SS2 + SS1) and return the
        SS2 reply to send back, or None if no SS1 was received."""
        ss2_reply = None
        for msg in self.messages:
            if len(msg) < 2:
                continue
            msg_type = msg[0]
            if msg_type == YMessageType.SYNC:
                reply = pycrdt.handle_sync_message(msg[1:], self.doc)
                if reply is not None:
                    # reply is an SS2 response to the server's SS1
                    ss2_reply = reply
        return ss2_reply

    def awareness_states(self) -> dict[int, dict]:
        """Decode every AwarenessUpdate the server sent to this client and
        return the resulting merged awareness states, keyed by client ID."""
        awareness = pycrdt.Awareness(Doc())
        for msg in self.messages:
            if len(msg) < 2 or msg[0] != YMessageType.AWARENESS:
                continue
            payload = pycrdt.read_message(msg[1:])
            awareness.apply_awareness_update(payload, origin=self)
        return awareness.states

    @property
    def source(self) -> str:
        return str(self.doc["source"])


class TestNormalSync:
    """Tests for normal (non-divergent) sync handshake."""

    @pytest.mark.asyncio
    async def test_fresh_client_syncs_successfully(self, make_yroom: MakeYRoom):
        """A fresh client with an empty YDoc should complete the handshake."""
        yroom = await make_yroom()
        ws = FakeWebSocket()
        client_id = yroom.clients.add(ws)

        # Send SS1 via add_message (goes through the queue)
        ss1 = ws.build_ss1()
        yroom.add_message(client_id, ss1)

        # Give the message queue time to process SS1 and await SS2
        await asyncio.sleep(0.1)

        # Client processes server's SS2 + SS1, gets SS2 reply
        ss2_reply = ws.process_server_messages()
        assert ss2_reply is not None

        # Send SS2 reply back (bypasses queue via future)
        yroom.add_message(client_id, ss2_reply)
        await asyncio.sleep(0.1)

        # Client should be synced
        client = yroom.clients.get(client_id)
        assert client.synced

    @pytest.mark.asyncio
    async def test_client_receives_existing_content(self, make_yroom: MakeYRoom):
        """A fresh client should receive the server's existing content."""
        yroom = await make_yroom()
        jupyter_ydoc = await yroom.get_jupyter_ydoc()
        jupyter_ydoc.source = "hello world"

        ws = FakeWebSocket()
        client_id = yroom.clients.add(ws)

        ss1 = ws.build_ss1()
        yroom.add_message(client_id, ss1)
        await asyncio.sleep(0.1)

        ss2_reply = ws.process_server_messages()
        assert ss2_reply is not None
        yroom.add_message(client_id, ss2_reply)
        await asyncio.sleep(0.1)

        # Client should have the server's content
        assert ws.source == "hello world"


class TestDivergentSync:
    """Tests for divergent client sync (content deduplication)."""

    @pytest.mark.asyncio
    async def test_divergent_client_detected(self, make_yroom: MakeYRoom):
        """A client with unknown client IDs should be detected as divergent."""
        yroom = await make_yroom()
        jupyter_ydoc = await yroom.get_jupyter_ydoc()
        jupyter_ydoc.source = "hello world"

        # Client has same content but different CRDT history
        ws = FakeWebSocket()
        ws.doc["source"] += "hello world"

        ss1 = ws.build_ss1()
        assert yroom._has_divergent_history(ss1[1:], yroom._ydoc.get_state())


class TestSyncTimeout:
    """Tests for handshake timeout behavior."""

    @pytest.mark.asyncio
    async def test_timeout_when_ss2_never_arrives(self, make_yroom: MakeYRoom):
        """If the client never sends SS2, the handshake should time out and
        the source should be restored."""
        yroom = await make_yroom()
        jupyter_ydoc = await yroom.get_jupyter_ydoc()
        jupyter_ydoc.source = "hello world"

        ws = FakeWebSocket()
        ws.doc["source"] += "hello world"
        client_id = yroom.clients.add(ws)

        ss1 = ws.build_ss1()
        yroom.add_message(client_id, ss1)

        # Wait for timeout (5s + buffer)
        await asyncio.sleep(6)

        # Source should be restored
        assert jupyter_ydoc.source == "hello world"
        # Saves should be re-enabled
        assert yroom.file_api._reloading_content is False
        # Buffer should be unpaused
        assert yroom.update_channel._paused is False


class TestLateSS2:
    """Regression tests for the blank-notebook data loss.

    A SyncStep2 reply arriving after `handshake_timeout` must be applied, and
    the client must not be disconnected. Dropping the late reply leaves the
    server ignorant of the client's IDs, so the client's next handshake is
    divergent again and the client-side repair then deletes the server's own
    content and syncs that deletion. See the `handle_sync` docstring.
    """

    @pytest.mark.asyncio
    async def test_late_ss2_is_applied(self, make_yroom: MakeYRoom):
        """An SS2 reply that misses the timeout is applied on arrival, and
        the client stays connected."""
        yroom = await make_yroom(handshake_timeout=0.3)
        jupyter_ydoc = await yroom.get_jupyter_ydoc()
        jupyter_ydoc.source = "hello "

        # Client with a local edit the server has never seen.
        ws = FakeWebSocket()
        ws.doc["source"] += "world"
        client_id = yroom.clients.add(ws)

        yroom.add_message(client_id, ws.build_ss1())

        # Let the handshake time out before the client replies. The unpaused
        # update channel proves the handshake window has genuinely closed, so
        # the reply below really does take the late path (guards against the
        # timeout kwarg being silently ignored, which would make this test
        # pass trivially via the in-time fast path).
        await asyncio.sleep(0.6)
        assert yroom.update_channel._paused is False

        # Client processes the server's SS2 + SS1 and replies -- late.
        ss2_reply = ws.process_server_messages()
        assert ss2_reply is not None
        yroom.add_message(client_id, ss2_reply)
        await asyncio.sleep(0.2)

        # The late reply must reach the server's YDoc...
        assert "world" in str(jupyter_ydoc.source)
        # ...and the client must still be connected.
        assert ws.closed is False
        assert yroom.clients.get(client_id) is not None

    @pytest.mark.asyncio
    async def test_timeout_does_not_disconnect(self, make_yroom: MakeYRoom):
        """Timeout ends the broadcast pause; it must not cut the client."""
        yroom = await make_yroom(handshake_timeout=0.3)
        ws = FakeWebSocket()
        client_id = yroom.clients.add(ws)

        yroom.add_message(client_id, ws.build_ss1())
        await asyncio.sleep(0.6)

        assert ws.closed is False
        assert yroom.clients.get(client_id) is not None
        assert yroom.update_channel._paused is False

    @pytest.mark.asyncio
    async def test_stale_ss2_during_another_clients_handshake(
        self, make_yroom: MakeYRoom
    ):
        """A late SS2 from client A arriving while client B's handshake is
        pending must still be applied, not dropped, and neither client may be
        disconnected."""
        yroom = await make_yroom(handshake_timeout=0.3)
        jupyter_ydoc = await yroom.get_jupyter_ydoc()
        jupyter_ydoc.source = "base "

        # A's handshake times out before A replies.
        ws_a = FakeWebSocket()
        ws_a.doc["source"] += "from-a"
        cid_a = yroom.clients.add(ws_a)
        yroom.add_message(cid_a, ws_a.build_ss1())
        await asyncio.sleep(0.6)
        ss2_a = ws_a.process_server_messages()
        assert ss2_a is not None

        # B starts a handshake; while it is pending, A's late reply arrives.
        ws_b = FakeWebSocket()
        cid_b = yroom.clients.add(ws_b)
        yroom.add_message(cid_b, ws_b.build_ss1())
        await asyncio.sleep(0.05)
        yroom.add_message(cid_a, ss2_a)

        # Complete B's handshake in time.
        ss2_b = ws_b.process_server_messages()
        assert ss2_b is not None
        yroom.add_message(cid_b, ss2_b)
        await asyncio.sleep(0.3)

        assert "from-a" in str(jupyter_ydoc.source)
        assert ws_a.closed is False
        assert ws_b.closed is False


async def _complete_handshake(yroom: YRoom, ws: FakeWebSocket) -> str:
    """Helper: add a FakeWebSocket client and complete the full sync handshake.
    Returns the client_id."""
    client_id = yroom.clients.add(ws)
    yroom.add_message(client_id, ws.build_ss1())
    await asyncio.sleep(0.1)
    ss2_reply = ws.process_server_messages()
    assert ss2_reply is not None, "Server did not send SS1 (no SS2 reply generated)"
    yroom.add_message(client_id, ss2_reply)
    await asyncio.sleep(0.1)
    return client_id


class TestSyncHandshakeStress:
    """
    Stress tests for data integrity when mutations occur during the sync
    handshake.

    These reproduce the scenario from jupyter-ai-contrib/jupyter-server-documents#197
    where an AI agent rapidly adds content via MCP tool calls while a second
    browser tab connects. Mutations that occur while a client is completing
    the handshake must not be lost.
    """

    @pytest.mark.asyncio
    async def test_mutations_before_handshake_not_lost(self, make_yroom: MakeYRoom):
        """Mutations between client connect and handshake must be received.

        Simulates: AI agent adds 20 lines while a second tab is connecting.
        """
        yroom = await make_yroom()
        jupyter_ydoc = await yroom.get_jupyter_ydoc()

        # Sync client A (first browser tab)
        ws_a = FakeWebSocket()
        await _complete_handshake(yroom, ws_a)

        # Client B connects (second browser tab) — starts as desynced
        ws_b = FakeWebSocket()
        cid_b = yroom.clients.add(ws_b)

        # While B is desynced, AI agent rapidly mutates the doc
        expected = ""
        for i in range(20):
            expected += f"AI added line {i}\n"
            jupyter_ydoc.source = expected

        # Complete B's handshake
        yroom.add_message(cid_b, ws_b.build_ss1())
        await asyncio.sleep(0.1)
        ss2_reply = ws_b.process_server_messages()
        assert ss2_reply is not None
        yroom.add_message(cid_b, ss2_reply)
        await asyncio.sleep(0.1)

        # B must have the full content — no data loss
        assert ws_b.source == expected

    @pytest.mark.asyncio
    async def test_mutations_during_handshake_await(self, make_yroom: MakeYRoom):
        """Mutations during the SS2 reply await must be received.

        Simulates: AI agent adds content while the server is waiting for the
        client's SS2 reply (the async gap in handle_sync).
        """
        yroom = await make_yroom()
        jupyter_ydoc = await yroom.get_jupyter_ydoc()
        jupyter_ydoc.source = "initial"

        # Sync client A
        ws_a = FakeWebSocket()
        await _complete_handshake(yroom, ws_a)

        # Client B starts handshake
        ws_b = FakeWebSocket()
        cid_b = yroom.clients.add(ws_b)
        yroom.add_message(cid_b, ws_b.build_ss1())
        await asyncio.sleep(0.1)
        # handle_sync is now awaiting B's SS2 reply

        # Mutate doc while handle_sync is awaiting
        jupyter_ydoc.source = "initial\nmutated during handshake"

        # Complete B's handshake
        ss2_reply = ws_b.process_server_messages()
        assert ss2_reply is not None
        yroom.add_message(cid_b, ss2_reply)
        await asyncio.sleep(0.1)

        # Broadcasts are paused during the handshake, so the mutation made while
        # the server awaited B's SS2 is delivered via the batched catchup diff
        # broadcast on resume. Process the post-handshake messages to apply it.
        ws_b.process_server_messages()

        assert ws_b.source == "initial\nmutated during handshake"

    @pytest.mark.asyncio
    async def test_no_exception_during_concurrent_handshakes(self, make_yroom: MakeYRoom):
        """Multiple clients handshaking while doc is mutated must not crash."""
        yroom = await make_yroom()
        jupyter_ydoc = await yroom.get_jupyter_ydoc()
        jupyter_ydoc.source = "initial"

        # Sync client A
        ws_a = FakeWebSocket()
        await _complete_handshake(yroom, ws_a)

        # Connect 5 desynced clients
        desynced = []
        for _ in range(5):
            ws = FakeWebSocket()
            cid = yroom.clients.add(ws)
            desynced.append((ws, cid))

        # Rapid mutations while all 5 are desynced
        for i in range(50):
            jupyter_ydoc.source = f"mutation {i}"

        # Sync all clients sequentially — no exceptions should be raised
        for ws, cid in desynced:
            yroom.add_message(cid, ws.build_ss1())
            await asyncio.sleep(0.1)
            ss2_reply = ws.process_server_messages()
            assert ss2_reply is not None
            yroom.add_message(cid, ss2_reply)
            await asyncio.sleep(0.1)

        # All must have the final state
        for ws, _ in desynced:
            assert ws.source == "mutation 49"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_mutations", [10, 50, 100])
    @pytest.mark.parametrize("num_clients", [2, 5])
    async def test_concurrent_mutations_stress(
        self, make_yroom: MakeYRoom, num_mutations: int, num_clients: int
    ):
        """N clients connect while the doc undergoes M mutations.
        All clients must converge to the same final state."""
        yroom = await make_yroom()
        jupyter_ydoc = await yroom.get_jupyter_ydoc()

        # Connect N desynced clients
        clients = []
        for _ in range(num_clients):
            ws = FakeWebSocket()
            cid = yroom.clients.add(ws)
            clients.append((ws, cid))

        # M mutations while all clients are desynced
        expected = ""
        for i in range(num_mutations):
            expected += f"line {i}\n"
            jupyter_ydoc.source = expected

        # Sync all clients
        for ws, cid in clients:
            yroom.add_message(cid, ws.build_ss1())
            await asyncio.sleep(0.1)
            ss2_reply = ws.process_server_messages()
            assert ss2_reply is not None
            yroom.add_message(cid, ss2_reply)
            await asyncio.sleep(0.1)

        # All must have the final content
        for ws, _ in clients:
            assert ws.source == expected


class TestSyncUpdateChannel:
    """Invariants on the update channel during the sync handshake."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "divergent", [False, True], ids=["normal", "divergent"]
    )
    async def test_update_channel_paused_during_sync(
        self, make_yroom: MakeYRoom, divergent: bool
    ):
        """The update channel must be paused for the duration of *any* sync
        handshake (normal or divergent), then resumed once it completes.

        The backend treats all syncs equivalently: broadcasts are paused so
        that mutations during the handshake gap are batched into a single
        catchup diff on resume rather than streamed individually (see
        YRoomUpdateChannel and #197).
        """
        yroom = await make_yroom()
        jupyter_ydoc = await yroom.get_jupyter_ydoc()
        jupyter_ydoc.source = "hello world"

        ws = FakeWebSocket()
        if divergent:
            # Same content, authored under a different client ID, so the
            # client's state vector contains an ID the server doesn't know.
            ws.doc["source"] += "hello world"
            assert yroom._has_divergent_history(
                ws.build_ss1()[1:], yroom._ydoc.get_state()
            )

        client_id = yroom.clients.add(ws)
        yroom.add_message(client_id, ws.build_ss1())
        await asyncio.sleep(0.1)

        # Mid-handshake: the server has sent SS2 + SS1 and is awaiting the
        # client's SS2 reply, so broadcasts must be paused.
        assert yroom.update_channel._paused is True

        ss2_reply = ws.process_server_messages()
        assert ss2_reply is not None
        yroom.add_message(client_id, ss2_reply)
        await asyncio.sleep(0.1)

        # Handshake complete: the channel must be resumed.
        assert yroom.update_channel._paused is False


def _seed_room_awareness(yroom: YRoom, **state) -> int:
    """Publish an awareness slot into the room from a simulated *other*
    client (e.g. a persona) and return that client's awareness ID.

    This mirrors how a peer publishes awareness before a new client connects:
    the peer encodes its local state and the room applies it via
    `handle_awareness_update`.
    """
    peer = pycrdt.Awareness(Doc())
    for field, value in state.items():
        peer.set_local_state_field(field, value)
    update = peer.encode_awareness_update([peer.client_id])
    yroom.handle_awareness_update("peer", pycrdt.create_awareness_message(update))
    return peer.client_id


class TestAwarenessOnConnect:
    """A newly-synced client must receive the room's *current* awareness state
    as part of the handshake, not only via later change deltas.

    Regression test for jupyter-ai-contrib/jupyter-server-documents#279: awareness
    published before a client connects (e.g. a persona) would only reach that
    client when a subsequent delta happened to re-touch the slot, causing
    seconds-long delays before personas appeared on refresh.
    """

    @pytest.mark.asyncio
    async def test_client_receives_existing_awareness_on_connect(
        self, make_yroom: MakeYRoom
    ):
        """A slot published *before* the client connects must be delivered by
        the handshake alone -- with no further awareness mutation."""
        yroom = await make_yroom()
        peer_id = _seed_room_awareness(yroom, name="Jupyternaut", type="persona")

        ws = FakeWebSocket()
        await _complete_handshake(yroom, ws)

        # The handshake alone (no later delta) must have delivered the slot.
        states = ws.awareness_states()
        assert peer_id in states
        assert states[peer_id] == {"name": "Jupyternaut", "type": "persona"}

    @pytest.mark.asyncio
    async def test_awareness_snapshot_only_to_new_client(
        self, make_yroom: MakeYRoom
    ):
        """The snapshot must go *only* to the newly-synced client, not be
        re-broadcast to peers already in the room."""
        yroom = await make_yroom()

        # An existing, already-synced client.
        ws_a = FakeWebSocket()
        await _complete_handshake(yroom, ws_a)

        _seed_room_awareness(yroom, name="Jupyternaut", type="persona")

        # Snapshot A's message count *after* the seed delta broadcast, right
        # before B connects. Any growth beyond this during B's handshake would
        # be a spurious room-wide re-broadcast of the connect snapshot.
        a_msgs_before_b = len(ws_a.messages)

        # A second client connects and completes the handshake.
        ws_b = FakeWebSocket()
        await _complete_handshake(yroom, ws_b)

        # The new client received the snapshot...
        assert any(
            len(m) >= 2 and m[0] == YMessageType.AWARENESS for m in ws_b.messages
        )
        # ...but the existing client received nothing extra from B's handshake.
        assert len(ws_a.messages) == a_msgs_before_b

    @pytest.mark.asyncio
    async def test_awareness_snapshot_sent_after_sync_step2(
        self, make_yroom: MakeYRoom
    ):
        """The snapshot must be sent *after* mark_synced, i.e. after the SS2
        reply that marks the client synced -- never before. Verify by message
        ordering: the AwarenessUpdate follows the first SYNC message."""
        yroom = await make_yroom()
        _seed_room_awareness(yroom, name="Jupyternaut", type="persona")

        ws = FakeWebSocket()
        await _complete_handshake(yroom, ws)

        first_sync_idx = next(
            i for i, m in enumerate(ws.messages)
            if len(m) >= 2 and m[0] == YMessageType.SYNC
        )
        awareness_idxs = [
            i for i, m in enumerate(ws.messages)
            if len(m) >= 2 and m[0] == YMessageType.AWARENESS
        ]
        assert awareness_idxs, "client never received an AwarenessUpdate"
        # Every awareness snapshot arrives after the SS2 sync reply that ran
        # mark_synced -- so a desynced client is never sent one.
        assert min(awareness_idxs) > first_sync_idx


class TestQueueRobustness:
    """No queued message may ever kill the room's message loop.

    A queued message can outlive its sender (tab closed while the message sat
    behind an in-flight handshake); processing it must not raise into
    `_process_message_queue`, or the room silently freezes for every client.
    """

    @pytest.mark.asyncio
    async def test_queued_update_from_removed_client_does_not_kill_loop(
        self, make_yroom: MakeYRoom
    ):
        yroom = await make_yroom()

        # Client A is synced and healthy.
        ws_a = FakeWebSocket()
        await _complete_handshake(yroom, ws_a)

        # A SyncUpdate arrives from a client that is no longer in the group
        # (its tab closed while the message was queued).
        ghost_doc = Doc()
        ghost_doc["source"] = Text()
        ghost_doc["source"] += "ghost edit"
        poison = pycrdt.create_update_message(ghost_doc.get_update())
        yroom.add_message("ghost-client-id", poison)
        await asyncio.sleep(0.1)

        # The loop must survive: a NEW client's handshake (which flows through
        # the same queue) must still complete.
        ws_b = FakeWebSocket()
        cid_b = yroom.clients.add(ws_b)
        yroom.add_message(cid_b, ws_b.build_ss1())
        await asyncio.sleep(0.1)
        ss2_reply = ws_b.process_server_messages()
        assert ss2_reply is not None, (
            "Message loop died processing a message from a removed client"
        )
        yroom.add_message(cid_b, ss2_reply)
        await asyncio.sleep(0.1)
        assert yroom.clients.get(cid_b).synced

    @pytest.mark.asyncio
    async def test_get_raises_keyerror_for_unknown_client(
        self, make_yroom: MakeYRoom
    ):
        yroom = await make_yroom()
        with pytest.raises(KeyError):
            yroom.clients.get("never-existed")


class TestDeadlineRace:
    """A SyncStep2 resolved in the same event-loop iteration as the handshake
    deadline must still be applied (Python >= 3.12 `wait_for` returns
    TimeoutError even when the awaited future holds a result)."""

    @pytest.mark.asyncio
    async def test_ss2_resolved_at_deadline_is_applied(
        self, make_yroom: MakeYRoom, monkeypatch: pytest.MonkeyPatch
    ):
        yroom = await make_yroom()

        # Force the race deterministically: for the pending-SS2 future only,
        # wait until the result has been delivered, then raise TimeoutError
        # anyway — exactly what CPython >= 3.12 does when the deadline timer
        # fires in the same iteration that resolved the future.
        real_wait_for = asyncio.wait_for

        async def racing_wait_for(awaitable, timeout=None):
            if any(awaitable is f for f in yroom._pending_ss2.values()):
                await asyncio.shield(awaitable)
                raise asyncio.TimeoutError()
            return await real_wait_for(awaitable, timeout)

        monkeypatch.setattr(asyncio, "wait_for", racing_wait_for)

        ws = FakeWebSocket()
        ws.doc["source"] += "client content that must survive the race"
        cid = yroom.clients.add(ws)
        yroom.add_message(cid, ws.build_ss1())
        await asyncio.sleep(0.1)
        ss2_reply = ws.process_server_messages()
        assert ss2_reply is not None
        yroom.add_message(cid, ss2_reply)
        await asyncio.sleep(0.2)

        ydoc = await yroom.get_ydoc()
        assert "client content that must survive the race" in str(ydoc["source"])
        # And the client must still be connected.
        assert not ws.closed


class TestStopDrain:
    """stop(immediately=False) must not lose queued messages or drop async
    teardown work."""

    @pytest.mark.asyncio
    async def test_stop_drain_applies_queued_ss2(self, make_yroom: MakeYRoom):
        yroom = await make_yroom()
        ws = FakeWebSocket()
        await _complete_handshake(yroom, ws)

        # The client makes an edit and answers a server SS1 with an SS2 that
        # is the sole carrier of that edit.
        ws.doc["source"] += "edit carried only by the SS2"
        ydoc = await yroom.get_ydoc()
        server_ss1 = pycrdt.create_sync_message(ydoc)
        ss2 = pycrdt.handle_sync_message(server_ss1[1:], ws.doc)
        assert ss2 is not None and ss2[1] == YSyncMessageSubtype.SYNC_STEP2

        # The SS2 sits in the queue when the room stops (no await between
        # enqueue and stop, so the message loop cannot consume it first).
        yroom._message_queue.put_nowait((yroom.clients.get_all()[0].id, ss2))
        yroom.stop()

        # The synchronous drain in stop() must have applied it to the YDoc.
        assert "edit carried only by the SS2" in str(ydoc["source"])

    @pytest.mark.asyncio
    async def test_stop_awaits_task_returning_callbacks(
        self, make_yroom: MakeYRoom
    ):
        yroom = await make_yroom()
        done = asyncio.Event()

        async def teardown():
            await asyncio.sleep(0.3)
            done.set()

        # Registration sites (session_manager, the execution handler) return
        # an already-scheduled Task from the callback, not a coroutine.
        yroom.add_stop_callback(lambda: asyncio.create_task(teardown()))

        yroom.stop()
        await yroom.until_saved
        assert done.is_set(), (
            "stop() dropped a Task returned by a stop callback; teardown "
            "raced shutdown"
        )

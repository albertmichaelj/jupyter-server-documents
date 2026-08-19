"""
Guards on the room/kernel lifecycle, added after the 2026-08 adversarial
review:

1. The room GC must not free a room with running or queued executions
   (the old guard read an awareness field nothing writes, so rooms were
   freeable mid-execution and the running work was cancelled).
2. A failed kernel connect must roll the room back to unwired — a sticky
   half-wired room made every later execute fail with "worker is not
   running" forever.
3. An unconfirmed shell (kernel busy at wire time) must be retried by the
   next execute instead of every execution being silently skipped.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from ...conftest import MakeYRoom


class _DeadKernelClient:
    """A kernel client whose heartbeat never answers."""

    def __init__(self, *args, **kwargs):
        self.hb_channel = SimpleNamespace(unpause=lambda: None)
        self.channels_stopped = False

    def load_connection_info(self, info):
        pass

    def start_channels(self):
        pass

    def stop_channels(self):
        self.channels_stopped = True

    async def _async_is_alive(self):
        return False


class _DeadKernelManager:
    client_factory = _DeadKernelClient

    def __init__(self):
        self.restart_callbacks = []

    def add_restart_callback(self, callback, event="restart"):
        self.restart_callbacks.append((callback, event))

    def remove_restart_callback(self, callback, event="restart"):
        self.restart_callbacks.remove((callback, event))

    def get_connection_info(self):
        return {"shell_port": 0}


class _SlowStartKernelManager(_DeadKernelManager):
    """A kernel manager whose kernel is COLD: the heartbeat answers only
    after `hb_alive` is set, so `connect_kernel` genuinely suspends in its
    heartbeat poll — the window in which a second connect can race it."""

    def __init__(self):
        super().__init__()
        self.hb_started = asyncio.Event()
        self.hb_alive = asyncio.Event()
        self.clients_stopped = 0
        manager = self

        class _Client(_DeadKernelClient):
            async def _async_is_alive(self):
                manager.hb_started.set()
                return manager.hb_alive.is_set()

            async def _async_wait_for_ready(self, *a, **k):
                return None

            def stop_channels(self):
                super().stop_channels()
                manager.clients_stopped += 1

        self.client_factory = _Client


class TestConnectSerialization:
    @pytest.mark.asyncio
    async def test_concurrent_connects_to_same_kernel_do_not_cross(
        self, make_yroom: MakeYRoom
    ):
        """create_session and the lazy re-wire can both call connect_kernel
        for the same cold kernel. Pre-lock, the second saw a client already
        assigned and disconnected it MID-SETUP — the worker then hit its
        kernel-client assertion and the user's run silently never happened
        (observed in fork CI's console-for-notebook integration test)."""
        room = await make_yroom(file_type="notebook")
        km = _SlowStartKernelManager()

        t1 = asyncio.create_task(room.connect_kernel(km))
        await asyncio.wait_for(km.hb_started.wait(), timeout=5)
        # Connect 1 is suspended in the heartbeat poll; the racing re-wire
        # arrives now.
        t2 = asyncio.create_task(room.connect_kernel(km))
        await asyncio.sleep(0.05)
        km.hb_alive.set()  # kernel heartbeat comes up
        await asyncio.wait_for(asyncio.gather(t1, t2), timeout=5)

        # Neither connect may destroy the other's client, and the room must
        # end fully wired with a live worker.
        assert km.clients_stopped == 0
        assert room.has_kernel_connection
        assert room._execution_queue is not None
        assert room._execution_worker_task is not None
        assert not room._execution_worker_task.done()


class TestGcExecutionGuard:
    @pytest.mark.asyncio
    async def test_gc_guard_blocks_running_cells(self, make_yroom: MakeYRoom):
        room = await make_yroom(file_type="notebook", inactivity_timeout=0)
        manager = room.parent
        await asyncio.sleep(0.05)

        # Control: idle and empty — freeable.
        assert manager._should_free_room(room) is True

        # A running cell (server-written awareness) must block freeing even
        # though the room is inactive and empty — this is the walked-away
        # long-computation case.
        room.set_cell_awareness_state("cell-1", "running")
        await asyncio.sleep(0.05)
        assert manager._should_free_room(room) is False

        room.set_cell_awareness_state("cell-1", "idle")
        await asyncio.sleep(0.05)
        assert manager._should_free_room(room) is True

    @pytest.mark.asyncio
    async def test_gc_guard_blocks_pending_executions(self, make_yroom: MakeYRoom):
        room = await make_yroom(file_type="notebook", inactivity_timeout=0)
        manager = room.parent
        await asyncio.sleep(0.05)

        room._execution_queue = asyncio.Queue()
        room._execution_queue.put_nowait(object())
        assert manager._should_free_room(room) is False

        room._execution_queue.get_nowait()
        room._worker_busy = True
        assert manager._should_free_room(room) is False

        room._worker_busy = False
        assert manager._should_free_room(room) is True


class TestConnectRollback:
    @pytest.mark.asyncio
    async def test_connect_kernel_failure_rolls_back(self, make_yroom: MakeYRoom):
        room = await make_yroom(file_type="notebook")
        room._heartbeat_timeout = 0.3
        km = _DeadKernelManager()

        with pytest.raises(RuntimeError):
            await room.connect_kernel(km)

        # The room must be fully unwired: a sticky _kernel_client would make
        # every later execute skip the re-wire and 400 forever.
        assert room.has_kernel_connection is False
        assert km.restart_callbacks == []

        # And a second attempt retries the wire from clean state.
        with pytest.raises(RuntimeError):
            await room.connect_kernel(km)
        assert room.has_kernel_connection is False


class TestShellConfirmedRetry:
    @pytest.mark.asyncio
    async def test_execute_cells_retries_kernel_info_when_unconfirmed(
        self, make_yroom: MakeYRoom, monkeypatch: pytest.MonkeyPatch
    ):
        room = await make_yroom(file_type="notebook")
        # Wired, worker running, but the readiness handshake never completed
        # (the kernel was busy when the lazy re-wire ran).
        room._kernel_client = object()
        room._execution_queue = asyncio.Queue()
        room._shell_confirmed = False

        calls = []

        async def fake_fetch():
            calls.append(1)
            room._shell_confirmed = True

        monkeypatch.setattr(room, "_fetch_kernel_info", fake_fetch)

        try:
            # The cell id doesn't exist; the retry hook is what this pins.
            await room.execute_cells([{"cell_id": "missing", "source_hash": "0"}])
        except Exception:
            pass

        assert calls == [1], (
            "execute_cells must re-attempt kernel-info when the shell was "
            "never confirmed; otherwise every execution is silently skipped"
        )


class TestResilientExecuteInteractive:
    @pytest.mark.asyncio
    async def test_spurious_empty_reenters_poll_loop(self, make_yroom: MakeYRoom):
        """The vendored execute loop must treat a spurious Empty from the
        iopub channel as a wakeup to re-poll — upstream jupyter_client lets
        it escape, killing an execution whose request was already sent (the
        kernel runs the cell; every output is silently lost)."""
        import zmq
        import zmq.asyncio
        from queue import Empty

        room = await make_yroom(file_type="notebook")

        ctx = zmq.asyncio.Context()
        recv_sock = ctx.socket(zmq.PULL)
        port = recv_sock.bind_to_random_port("tcp://127.0.0.1")
        send_sock = ctx.socket(zmq.PUSH)
        send_sock.connect(f"tcp://127.0.0.1:{port}")

        msg_id = "req-1"

        def iopub(msg_type, content):
            return {
                "parent_header": {"msg_id": msg_id},
                "header": {"msg_type": msg_type},
                "content": content,
            }

        script = [
            "SPURIOUS-EMPTY",
            iopub("status", {"execution_state": "busy"}),
            iopub("execute_result", {"data": {"text/plain": "2"}}),
            iopub("status", {"execution_state": "idle"}),
        ]
        empties = []

        class _IopubChannel:
            socket = recv_sock

            def is_alive(self):
                return True

            async def get_msg(self, timeout=0):
                await recv_sock.recv()  # consume one wakeup
                entry = script.pop(0)
                if entry == "SPURIOUS-EMPTY":
                    empties.append(1)
                    raise Empty
                return entry

        class _Client:
            iopub_channel = _IopubChannel()

            def execute(self, code, allow_stdin=False):
                return msg_id

            async def _async_recv_reply(self, mid, timeout=None):
                return {"content": {"status": "ok"}}

        room._kernel_client = _Client()
        try:
            # One wakeup per scripted get_msg call.
            for _ in range(len(script)):
                await send_sock.send(b"x")

            outputs = []
            reply = await asyncio.wait_for(
                room._execute_interactive("1 + 1", outputs.append), timeout=10
            )

            assert empties, "the spurious-Empty branch was never exercised"
            assert any(
                m["header"]["msg_type"] == "execute_result" for m in outputs
            ), "outputs after the spurious Empty were lost"
            assert reply["content"]["status"] == "ok"
        finally:
            recv_sock.close(0)
            send_sock.close(0)
            ctx.term()

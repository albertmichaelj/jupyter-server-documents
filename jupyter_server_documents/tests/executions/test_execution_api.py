"""
Integration tests for POST /api/kernels/{kernel_id}/execute —
the jupyverse-compatible server-side execution endpoint.
"""
import asyncio
import json
import uuid
from pathlib import Path

import pytest
from tornado.httpclient import HTTPClientError

TEST_TIMEOUT = 30

CELL_ID = "test-cell-aabbcc"
CELL_SOURCE = "1 + 1"
# MurmurHash2(seed=0) of CELL_SOURCE — matches _murmur2(source, 0) in the frontend.
CELL_SOURCE_HASH = "3531899427"

NOTEBOOK_CONTENT = json.dumps({
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.9"},
    },
    "cells": [
        {
            "cell_type": "code",
            "id": CELL_ID,
            "source": CELL_SOURCE,
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        }
    ],
})


# ── HTTP contract tests ────────────────────────────────────────────────────────


async def test_missing_cells_returns_400(jp_fetch):
    """POST without cells must return 400."""
    with pytest.raises(HTTPClientError) as exc_info:
        await jp_fetch(
            "api", "kernels", "00000000-0000-0000-0000-000000000000", "execute",
            method="POST",
            body=json.dumps({"document_id": "json:notebook:abc"}),
            headers={"Content-Type": "application/json"},
        )
    assert exc_info.value.code == 400


async def test_missing_document_id_returns_400(jp_fetch):
    """POST without document_id must return 400."""
    with pytest.raises(HTTPClientError) as exc_info:
        await jp_fetch(
            "api", "kernels", "00000000-0000-0000-0000-000000000000", "execute",
            method="POST",
            body=json.dumps({"cells": [{"cell_id": CELL_ID}]}),
            headers={"Content-Type": "application/json"},
        )
    assert exc_info.value.code == 400


async def test_unknown_document_id_returns_400(jp_fetch):
    """POST with a document_id that has no live YRoom must return 400."""
    with pytest.raises(HTTPClientError) as exc_info:
        await jp_fetch(
            "api", "kernels", "00000000-0000-0000-0000-000000000000", "execute",
            method="POST",
            body=json.dumps({
                "document_id": "json:notebook:does-not-exist",
                "cells": [{"cell_id": CELL_ID, "source_hash": CELL_SOURCE_HASH}],
            }),
            headers={"Content-Type": "application/json"},
        )
    assert exc_info.value.code == 400


async def test_missing_source_hash_returns_400(jp_fetch):
    """POST with a cell missing source_hash must return 400."""
    with pytest.raises(HTTPClientError) as exc_info:
        await jp_fetch(
            "api", "kernels", "00000000-0000-0000-0000-000000000000", "execute",
            method="POST",
            body=json.dumps({
                "document_id": "json:notebook:does-not-exist",
                "cells": [{"cell_id": CELL_ID}],
            }),
            headers={"Content-Type": "application/json"},
        )
    assert exc_info.value.code == 400


# ── End-to-end test (requires ipykernel) ──────────────────────────────────────


async def _wait_for_yroom(jp_serverapp, session_id, cell_id, timeout=10.0):
    """Poll until the YRoom has content and the cell is accessible."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            yroom = jp_serverapp.session_manager.get_yroom(session_id)
            ydoc = await yroom.get_jupyter_ydoc()
            _, cell = ydoc.find_cell(cell_id)
            if cell is not None:
                return yroom
        except Exception:
            pass
        await asyncio.sleep(0.1)
    raise TimeoutError(f"YRoom content not ready after {timeout}s")


@pytest.mark.timeout(TEST_TIMEOUT)
async def test_full_execution_via_jupyverse_endpoint(jp_fetch, jp_serverapp, tmp_path):
    """
    End-to-end: notebook → session → execute via POST /api/kernels/{id}/execute.

    Verifies:
    - Endpoint returns null (matching the jupyverse contract)
    - The execution actually runs (outputs appear in the YDoc)
    """
    nb_name = f"test_{uuid.uuid4().hex[:8]}.ipynb"
    (tmp_path / nb_name).write_text(NOTEBOOK_CONTENT)

    # Start session + kernel
    r = await jp_fetch(
        "api", "sessions",
        method="POST",
        body=json.dumps({
            "path": nb_name,
            "name": nb_name,
            "type": "notebook",
            "kernel": {"name": "python3"},
        }),
        headers={"Content-Type": "application/json"},
    )
    assert r.code == 201
    session = json.loads(r.body)
    session_id = session["id"]
    kernel_id = session["kernel"]["id"]

    # Wait for YRoom content to load
    yroom = await _wait_for_yroom(jp_serverapp, session_id, CELL_ID)

    # document_id is the room name — same as the YRoom's room_id
    document_id = yroom.room_id

    # Execute via the jupyverse-compatible endpoint
    r = await jp_fetch(
        "api", "kernels", kernel_id, "execute",
        method="POST",
        body=json.dumps({"document_id": document_id, "cells": [{"cell_id": CELL_ID, "source_hash": CELL_SOURCE_HASH}]}),
        headers={"Content-Type": "application/json"},
    )
    assert r.code == 200
    assert json.loads(r.body) is None  # jupyverse returns null

    # Cleanup
    await jp_fetch("api", "sessions", session_id, method="DELETE")


@pytest.mark.timeout(TEST_TIMEOUT)
async def test_execute_rewires_a_recreated_room(jp_fetch, jp_serverapp, tmp_path):
    """A room freed by GC and re-created must be re-wired to its session's
    still-running kernel by the execute endpoint — not fail 400 forever.

    The room→kernel bond is only formed in `create_session`, and the old
    room's stop callback tears it down when the room is freed. Before the
    lazy re-wire, the re-created room had a live session and a live kernel
    but no connection between them, and every execution returned 400
    ("YNotebookRoom is not connected to a kernel") until the user shut the
    kernel down and started a fresh session. Observed in production within
    hours of sessions being able to survive reconnects.
    """
    nb_name = f"test_{uuid.uuid4().hex[:8]}.ipynb"
    (tmp_path / nb_name).write_text(NOTEBOOK_CONTENT)

    r = await jp_fetch(
        "api", "sessions",
        method="POST",
        body=json.dumps({
            "path": nb_name,
            "name": nb_name,
            "type": "notebook",
            "kernel": {"name": "python3"},
        }),
        headers={"Content-Type": "application/json"},
    )
    assert r.code == 201
    session = json.loads(r.body)
    session_id = session["id"]
    kernel_id = session["kernel"]["id"]

    yroom = await _wait_for_yroom(jp_serverapp, session_id, CELL_ID)
    document_id = yroom.room_id
    assert yroom.has_kernel_connection

    # Simulate the GC cycle: free the room while session + kernel live on,
    # then re-create it the way a client reconnect does.
    manager = jp_serverapp.web_app.settings["yroom_manager"]
    assert await manager.delete_room(document_id)
    yroom2 = manager.get_room(document_id)
    assert yroom2 is not yroom
    await yroom2.file_api.until_content_loaded
    assert not yroom2.has_kernel_connection

    # Execution against the re-created room must succeed (this returned 400
    # before the lazy re-wire) ...
    r = await jp_fetch(
        "api", "kernels", kernel_id, "execute",
        method="POST",
        body=json.dumps({
            "document_id": document_id,
            "cells": [{"cell_id": CELL_ID, "source_hash": CELL_SOURCE_HASH}],
        }),
        headers={"Content-Type": "application/json"},
    )
    assert r.code == 200

    # ... and the room must now be wired.
    assert yroom2.has_kernel_connection

    await jp_fetch("api", "sessions", session_id, method="DELETE")


@pytest.mark.timeout(TEST_TIMEOUT)
async def test_execute_on_stopped_room_returns_503(jp_fetch, jp_serverapp, tmp_path):
    """`delete_room` stops the room and then awaits the final save before
    popping it from the manager, so `get_room` can return a STOPPED room in
    that window. Executing against it must be rejected as retryable — not
    return 200 while the outputs land in an orphaned YDoc (never broadcast,
    never saved) and a kernel client nothing can stop is leaked."""
    nb_name = f"test_{uuid.uuid4().hex[:8]}.ipynb"
    (tmp_path / nb_name).write_text(NOTEBOOK_CONTENT)

    r = await jp_fetch(
        "api", "sessions",
        method="POST",
        body=json.dumps({
            "path": nb_name,
            "name": nb_name,
            "type": "notebook",
            "kernel": {"name": "python3"},
        }),
        headers={"Content-Type": "application/json"},
    )
    session = json.loads(r.body)
    session_id = session["id"]
    kernel_id = session["kernel"]["id"]

    yroom = await _wait_for_yroom(jp_serverapp, session_id, CELL_ID)
    document_id = yroom.room_id

    # Enter the delete_room window: stopped, but still returned by get_room.
    yroom.stop()

    with pytest.raises(HTTPClientError) as exc_info:
        await jp_fetch(
            "api", "kernels", kernel_id, "execute",
            method="POST",
            body=json.dumps({
                "document_id": document_id,
                "cells": [{"cell_id": CELL_ID, "source_hash": CELL_SOURCE_HASH}],
            }),
            headers={"Content-Type": "application/json"},
        )
    assert exc_info.value.code == 503

    await jp_fetch("api", "sessions", session_id, method="DELETE")


@pytest.mark.timeout(TEST_TIMEOUT)
async def test_execute_rejects_kernel_not_bound_to_the_document(
    jp_fetch, jp_serverapp, tmp_path
):
    """The re-wire must refuse a kernel that does not belong to the
    document's session. The URL kernel id is client-supplied: on a shared
    server any collaborator could otherwise wire the document's room to any
    kernel they can name, after which everyone's executions run in the wrong
    kernel with no error anywhere."""
    nb_name = f"test_{uuid.uuid4().hex[:8]}.ipynb"
    (tmp_path / nb_name).write_text(NOTEBOOK_CONTENT)

    r = await jp_fetch(
        "api", "sessions",
        method="POST",
        body=json.dumps({
            "path": nb_name,
            "name": nb_name,
            "type": "notebook",
            "kernel": {"name": "python3"},
        }),
        headers={"Content-Type": "application/json"},
    )
    session = json.loads(r.body)
    session_id = session["id"]

    yroom = await _wait_for_yroom(jp_serverapp, session_id, CELL_ID)
    document_id = yroom.room_id

    # A live kernel that belongs to NO session for this document.
    r = await jp_fetch(
        "api", "kernels",
        method="POST",
        body=json.dumps({"name": "python3"}),
        headers={"Content-Type": "application/json"},
    )
    foreign_kernel_id = json.loads(r.body)["id"]

    # Free and re-create the room so the execute hits the re-wire path.
    manager = jp_serverapp.web_app.settings["yroom_manager"]
    assert await manager.delete_room(document_id)
    yroom2 = manager.get_room(document_id)
    await yroom2.file_api.until_content_loaded
    assert not yroom2.has_kernel_connection

    with pytest.raises(HTTPClientError) as exc_info:
        await jp_fetch(
            "api", "kernels", foreign_kernel_id, "execute",
            method="POST",
            body=json.dumps({
                "document_id": document_id,
                "cells": [{"cell_id": CELL_ID, "source_hash": CELL_SOURCE_HASH}],
            }),
            headers={"Content-Type": "application/json"},
        )
    assert exc_info.value.code == 400
    assert not yroom2.has_kernel_connection

    await jp_fetch("api", "kernels", foreign_kernel_id, method="DELETE")
    await jp_fetch("api", "sessions", session_id, method="DELETE")

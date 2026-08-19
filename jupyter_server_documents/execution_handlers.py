import asyncio

from jupyter_server.auth.decorator import authorized
from jupyter_server.base.handlers import APIHandler
from tornado import web
from tornado.escape import json_encode

from .rooms.ynotebook_room import YNotebookRoom, SourceMismatchError, PredecessorTimeoutError


AUTH_RESOURCE = "executions"


class ExecutionsAPIHandler(APIHandler):
    auth_resource = AUTH_RESOURCE


class KernelExecuteHandler(ExecutionsAPIHandler):
    """
    POST /api/kernels/{kernel_id}/execute

    Server-side cell execution endpoint.

    ## Request body

    ```json
    {
      "document_id": "string",   // required — room name
      "cells": [                 // required — cells to execute atomically and in order
        {
          "cell_id":     "string",  // required — cell ID
          "source_hash": "string"   // required — MurmurHash2 (seed=0) of cell source, as decimal string
        }
      ],

      // Execution ordering (optional)
      "client_id":          "string",  // document client ID
      "request_id":         "string",  // UUID for this request
      "previous_request_id":"string"   // wait for this request to be enqueued first
    }
    ```

    All cells in ``cells`` are verified (hash check) and enqueued atomically
    before the response is sent, so no other request can interleave with the
    batch.  This makes "Run All" and "Restart and Run All" safe regardless of
    network timing.

    The ``source_hash`` per cell is a MurmurHash2 (seed=0) decimal string of
    the cell source at the time the user pressed Run.  The server returns 409
    if the YDoc source has diverged (another user edited the cell after the
    request was sent).

    ## Responses
    - ``200 null``  — accepted (fire-and-forget)
    - ``400``       — bad request
    - ``408``       — predecessor request timed out
    - ``409 {"error": "source_mismatch", "cell_id": "..."}`` — source diverged
    """

    @web.authenticated
    @authorized
    async def post(self, kernel_id: str):
        body = self.get_json_body() or {}
        document_id = body.get("document_id")

        if not document_id:
            raise web.HTTPError(400, "document_id is required")

        cells_payload = body.get("cells")
        if not cells_payload or not isinstance(cells_payload, list):
            raise web.HTTPError(400, "cells must be a non-empty list of {cell_id, source_hash}")

        client_id = body.get("client_id")
        request_id = body.get("request_id")
        previous_request_id = body.get("previous_request_id")

        self.log.info(
            "execute POST: kernel=%s document=%r cells=%d",
            kernel_id,
            document_id,
            len(cells_payload),
        )
        yroom = self.settings["yroom_manager"].get_room(document_id)
        if yroom is None:
            raise web.HTTPError(400, f"No YRoom available for document: {document_id!r}")
        if not isinstance(yroom, YNotebookRoom):
            raise web.HTTPError(400, f"Room {document_id!r} is not a notebook room")
        if yroom.stopped:
            # delete_room stops the room and then awaits the final save before
            # popping it from the manager, so get_room can return a stopped
            # room in that window. Wiring or enqueueing into it would return
            # 200 while the outputs land in an orphaned YDoc (never broadcast,
            # never saved) and would leak a kernel client nothing can stop.
            raise web.HTTPError(503, f"Room {document_id!r} is shutting down; retry")

        if not yroom.has_kernel_connection:
            # A room that was garbage-collected and later re-created has a live
            # session but no kernel wiring: the room→kernel bond is only formed
            # in `create_session`, and the old room's stop callback tore it
            # down. Without this, every execution against the re-created room
            # fails ("YNotebookRoom is not connected to a kernel") until the
            # user shuts the kernel down and starts a fresh session. The
            # request names the kernel in the URL, so re-wire here exactly the
            # way `create_session` does.
            try:
                kernel_manager = self.kernel_manager.get_kernel(kernel_id)
            except (KeyError, web.HTTPError):
                # MappingKernelManager raises HTTPError(404) itself for an
                # unknown kernel; older managers raise KeyError.
                # 400 (not 404) to preserve the endpoint's existing contract:
                # before the lazy re-wire, this request shape fell through to
                # execute_cells' RuntimeError and returned 400.
                raise web.HTTPError(400, f"Kernel not found: {kernel_id}")

            # The URL kernel id is client-supplied; validate it against the
            # document's session before forming a room-wide binding. On a
            # shared server any collaborator could otherwise wire this
            # document's room to any kernel they can name — and a stale tab
            # could wire it to a replaced kernel — after which everyone's
            # executions run in the wrong kernel with no error anywhere.
            file_id = yroom.room_id.split(":", 2)[2]
            fim = self.settings.get("file_id_manager")
            doc_path = fim.get_path(file_id) if fim is not None else None
            sessions = await self.settings["session_manager"].list_sessions()
            session_kernel_ids = {
                (s.get("kernel") or {}).get("id")
                for s in sessions
                if doc_path is not None and s.get("path") == doc_path
            }
            if kernel_id not in session_kernel_ids:
                raise web.HTTPError(
                    400,
                    f"Kernel {kernel_id} does not belong to the session for "
                    f"document {document_id!r}",
                )

            try:
                await yroom.connect_kernel(kernel_manager)
            except Exception as e:
                # connect_kernel rolled the room back to unwired, so the next
                # execute retries the wire; 503 marks this retryable rather
                # than leaving a half-wired room that 400s forever.
                self.log.warning(
                    "Failed to reconnect room %r to kernel %s: %s",
                    document_id,
                    kernel_id,
                    e,
                )
                raise web.HTTPError(503, f"Kernel {kernel_id} is not responding; retry")
            if yroom.stopped:
                # The room stopped while we were connecting; its stop
                # callbacks have already run, so ours would never fire.
                await yroom.disconnect_kernel()
                raise web.HTTPError(503, f"Room {document_id!r} is shutting down; retry")
            yroom.add_stop_callback(
                lambda: asyncio.create_task(yroom.disconnect_kernel())
            )
            self.log.info(
                "Reconnected room %r to kernel %s for execution.",
                document_id,
                kernel_id,
            )

        try:
            await yroom.execute_cells(
                cells_payload,
                clear_outputs=True,
                request_id=request_id,
                previous_request_id=previous_request_id,
                client_id=client_id,
            )
        except SourceMismatchError as e:
            self.set_status(409)
            self.finish(json_encode({"error": "source_mismatch", "cell_id": e.cell_id}))
            return
        except PredecessorTimeoutError:
            raise web.HTTPError(408, "Timed out waiting for previous_request_id to be enqueued")
        except (LookupError, ValueError, RuntimeError) as e:
            raise web.HTTPError(400, str(e))

        self.finish("null")


executions_handlers = [
    (r"api/kernels/([\w-]+)/execute", KernelExecuteHandler),
]

from pycrdt import Map

from traitlets import Bool, Set
from traitlets.config import LoggingConfigurable


class OutputProcessor(LoggingConfigurable):
    """
    Writes kernel output messages into a live pycrdt YDoc cell.

    All methods take a direct `ycell` reference (pycrdt.Map) and write
    synchronously.  There is no async work — the old async lookup chain
    (session → file_id → room → find_cell) is gone because the caller
    already has ycell, file_id, and cell_id.

    Writing synchronously also eliminates the race condition where a
    create_task from a previous execution could run after the cell was
    cleared for re-execution, appending stale outputs.
    """

    _pending_clear_output_cells: Set = Set(default_value=set())

    use_outputs_service = Bool(
        default_value=True,
        help="Route outputs through the outputs service to minimise in-memory YDoc size.",
    ).tag(config=True)

    @property
    def outputs_manager(self):
        return self.parent.parent.parent.parent.web_app.settings["outputs_manager"]

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_output(
        self,
        msg_type: str,
        ycell,
        file_id: str | None,
        cell_id: str,
        content: dict,
    ) -> None:
        """
        Write an output message into the YDoc cell synchronously.

        Synchronous execution prevents the race where a stale task from a
        previous execution appends outputs after the cell has been cleared.
        """
        if msg_type == "clear_output":
            self._handle_clear_output(ycell, file_id, cell_id, content)
        else:
            self._write_output(msg_type, ycell, file_id, cell_id, content)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _handle_clear_output(self, ycell, file_id: str | None, cell_id: str, content: dict):
        wait = content.get("wait", False)
        if wait:
            self._pending_clear_output_cells.add(cell_id)
        else:
            self._clear_ycell_outputs(ycell, file_id, cell_id)

    def _write_output(
        self,
        msg_type: str,
        ycell,
        file_id: str | None,
        cell_id: str,
        content: dict,
    ):
        # Flush any pending clear before appending new output
        if cell_id in self._pending_clear_output_cells:
            self._clear_ycell_outputs(ycell, file_id, cell_id)
            self._pending_clear_output_cells.discard(cell_id)

        display_id = content.get("transient", {}).get("display_id")

        if self.use_outputs_service and file_id:
            output = self.transform_output(msg_type, content, ydoc=False)
            output = self.outputs_manager.write(
                file_id=file_id,
                cell_id=cell_id,
                output=output,
                display_id=display_id,
            )
        else:
            output = self.transform_output(msg_type, content, ydoc=False)

        if output is None:
            return

        output_index = (
            self.outputs_manager.get_output_index(display_id)
            if display_id and self.use_outputs_service else None
        )
        # This output came from the user's own kernel, in a session the user
        # started, so it is trusted -- mark it before it lands in the cell.
        self._mark_trusted(ycell)

        outputs = ycell["outputs"]
        if output_index is not None and output_index < len(outputs):
            outputs[output_index] = output
        else:
            if output_index is not None:
                self.log.warning(
                    f"Stale output index {output_index} for display_id {display_id!r} "
                    f"(outputs length: {len(outputs)}), appending instead."
                )
            outputs.append(output)

    def _mark_trusted(self, ycell) -> None:
        """Mark a cell's outputs as trusted.

        WHY THIS EXISTS. Jupyter's trust model asks one question of a rich
        output: did this user's own kernel produce it, in a session this user
        started? In classic Jupyter the FRONTEND answers it -- outputs arrive
        over the kernel websocket it opened, so it marks them trusted. Here the
        server writes outputs straight into the shared document and the
        frontend only sees a document change, with no provenance attached, so
        nobody was answering the question at all.

        The consequences were quiet and confusing. `NotebookNotary._check_cell`
        refuses to trust any cell holding an `execute_result` or `display_data`
        with data keys unless `metadata.trusted` is set, and
        `ContentsManager.save` only signs a notebook when EVERY cell passes --
        logging "Notebook X is not trusted" otherwise. An unsigned notebook has
        its HTML output sanitized in the browser, which strips `<style>`. That
        is why a scikit-learn or XGBoost estimator repr rendered as its
        plain-text fallback plus an unstyled list of parameters instead of the
        collapsible diagram: the markup survived, the CSS did not.

        Setting the flag here is the same answer classic Jupyter gives, at the
        only place that knows it: output the server just produced. It does NOT
        trust output that was already in the file when the room loaded -- those
        cells keep whatever trust they arrived with, so a downloaded notebook's
        saved outputs are still sanitized until re-executed.

        `jupyter_ydoc` sets this same field on the cells it creates
        ("auto-created empty code cell without outputs ought be trusted"), so
        the key round-trips through the shared model and out to nbformat.

        Wrapped defensively: a cell that renders its output is worth more than
        one that is correctly trusted, so a metadata quirk must never take the
        output path down with it.
        """
        try:
            metadata = ycell.get("metadata")
            if metadata is None:
                ycell["metadata"] = {"trusted": True}
            elif not metadata.get("trusted", False):
                metadata["trusted"] = True
        except Exception:
            self.log.debug("Could not mark cell as trusted.", exc_info=True)

    def _clear_ycell_outputs(self, ycell, file_id: str | None, cell_id: str):
        del ycell["outputs"][:]
        if self.use_outputs_service and file_id:
            self.outputs_manager.clear(file_id=file_id, cell_id=cell_id)

    # ── Output transformation ──────────────────────────────────────────────────

    def transform_output(self, msg_type: str, content: dict, ydoc: bool = False):
        """Convert an iopub message content dict to nbformat output structure."""
        factory = Map if ydoc else (lambda x: x)
        if msg_type == "stream":
            return factory({
                "output_type": "stream",
                "text": content["text"],
                "name": content["name"],
            })
        if msg_type in ("display_data", "update_display_data"):
            return factory({
                "output_type": "display_data",
                "data": content["data"],
                "metadata": content["metadata"],
            })
        if msg_type == "execute_result":
            return factory({
                "output_type": "execute_result",
                "data": content["data"],
                "metadata": content["metadata"],
                "execution_count": content["execution_count"],
            })
        if msg_type == "error":
            return factory({
                "output_type": "error",
                "traceback": content["traceback"],
                "ename": content["ename"],
                "evalue": content["evalue"],
            })
        return None

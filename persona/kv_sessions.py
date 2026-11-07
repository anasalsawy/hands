"""Durable llama-server KV checkpoints for scoped Persona generation."""
from __future__ import annotations

from contextlib import aclosing, asynccontextmanager
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

log = logging.getLogger(__name__)


class KVCapacityError(RuntimeError):
    pass


class KVSlots:
    def __init__(self, directory: Path, slots: int, epoch: str):
        if slots < 1 or not epoch:
            raise ValueError("KV slots and model/config epoch are required")
        self.directory = directory
        self.epoch = epoch
        self.owners: list[str | None] = [None] * slots
        self.busy: set[int] = set()
        self.active: set[str] = set()
        self.boot_ids: list[str | None] = [None] * slots

    @asynccontextmanager
    async def use(self, llm, scope, model):
        key = sha256(json.dumps([self.epoch, model, *scope]).encode()).hexdigest()
        available = [i for i in range(len(self.owners)) if i not in self.busy]
        if not available or key in self.active:
            raise KVCapacityError("No inference KV slot is available for this session")
        slot = min(available, key=lambda i: (self.owners[i] != key, self.owners[i] is not None))
        self.busy.add(slot)
        self.active.add(key)
        manifest = self.directory / (key + ".json")
        previous = None
        try:
            try:
                previous = manifest.read_text().strip()
                if not previous.startswith(key + "-") or Path(previous).name != previous:
                    previous = None
            except FileNotFoundError:
                pass
            try:
                boot_id = (self.directory / "engine-boot-id").read_text().strip()
            except FileNotFoundError:
                boot_id = ""
            if self.owners[slot] != key or not boot_id or self.boot_ids[slot] != boot_id:
                await self._action(llm, slot, "erase")
                if previous:
                    try:
                        receipt = await self._action(llm, slot, "restore", previous)
                        llm.call_stats["kv_restored_tokens"] = int(receipt["n_restored"])
                    except Exception:
                        log.warning("KV checkpoint restore failed; rebuilding from the transcript")
                        await self._action(llm, slot, "erase")
                self.owners[slot] = key
                self.boot_ids[slot] = boot_id
            yield slot
            filename = key + "-" + uuid4().hex + ".bin"
            temporary = self.directory / (filename + ".tmp")
            checkpoint = self.directory / filename
            try:
                receipt = await self._action(llm, slot, "save", filename)
                count = int(receipt["n_saved"])
                if count <= 0:
                    raise ValueError("Engine saved an empty KV checkpoint")
                with checkpoint.open("rb") as handle:
                    os.fsync(handle.fileno())
                with temporary.open("x") as handle:
                    handle.write(filename)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, manifest)
                fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                llm.call_stats["kv_checkpoint_tokens"] = count
                if previous:
                    (self.directory / previous).unlink(missing_ok=True)
            except Exception:
                log.warning("Could not publish inference KV checkpoint", exc_info=True)
                temporary.unlink(missing_ok=True)
        except BaseException:
            self.owners[slot] = None
            raise
        finally:
            self.busy.remove(slot)
            self.active.remove(key)

    async def _action(self, llm, slot, action, filename=None):
        root = llm.endpoint.base_url.rstrip("/").removesuffix("/v1")
        response = await llm._http().post(
            f"{root}/slots/{slot}",
            params={"action": action},
            headers=llm._headers(),
            json={"filename": filename} if filename else {},
        )
        response.raise_for_status()
        return response.json()


class KVSessionLLM:
    """Bind a logical lane to a dedicated llama-server without queuing."""

    def __init__(self, engine, fallback, slots: KVSlots, scope):
        self.engine = engine
        self.fallback = fallback
        self.slots = slots
        self.scope = scope

    def __getattr__(self, name):
        return getattr(self.engine, name)

    async def chat(self, messages, **options):
        try:
            async with self.slots.use(self.engine, self.scope, self.engine.model) as slot:
                return await self.engine.chat(messages, **{**options, "model": None, "_kv_slot": slot})
        except KVCapacityError:
            log.warning("Inference KV capacity exhausted; using the configured fallback")
            return await self.fallback.chat(messages, **options)

    async def stream(self, messages, **options):
        try:
            async with self.slots.use(self.engine, self.scope, self.engine.model) as slot:
                stream = self.engine.stream(messages, **{**options, "model": None, "_kv_slot": slot})
                terminal = []
                async with aclosing(stream):
                    async for event in stream:
                        if event.get("type") in ("tool_call", "done"):
                            terminal.append(event)
                        else:
                            yield event
            for event in terminal:
                yield event
        except KVCapacityError:
            log.warning("Inference KV capacity exhausted; using the configured fallback")
            async with aclosing(self.fallback.stream(messages, **options)) as stream:
                async for event in stream:
                    yield event

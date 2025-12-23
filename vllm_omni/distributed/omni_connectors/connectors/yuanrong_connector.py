# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Yuanrong OmniConnector implementation using openYuanrong kvclient.

This connector mirrors the behavior of :class:`MooncakeConnector` but relies on
the kvclient package from the openYuanrong data system for transporting data
between pipeline stages. Objects are serialized through the shared
``OmniSerializer`` utilities provided by :class:`~.base.OmniConnectorBase`.
"""

from __future__ import annotations

import importlib
import time
from typing import Any, Callable, Optional

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

kvclient_spec = importlib.util.find_spec("kvclient")
if kvclient_spec is not None:
    kvclient = importlib.import_module("kvclient")  # type: ignore
else:
    logger.warning("kvclient not available, YuanrongConnector will not work")
    kvclient = None


class YuanrongConnector(OmniConnectorBase):
    """kvclient-based distributed connector for OmniConnector."""

    def __init__(self, config: dict[str, Any]):
        if kvclient is None:
            raise ImportError("kvclient not available")

        self.config = config
        self.endpoints = config.get("endpoints") or []
        self.namespace = config.get("namespace")
        self.username = config.get("username")
        self.password = config.get("password")
        self.client_kwargs = config.get("client_kwargs", {})

        self.client = self._init_client()

        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "errors": 0,
            "timeouts": 0,
        }

    def _make_key(self, rid: str, from_stage: str, to_stage: str) -> str:
        return f"{rid}/{from_stage}_to_{to_stage}"

    def _init_client(self):
        """Instantiate a kvclient using available constructor signatures."""

        def _instantiate(ctor: Callable[..., Any], kwargs: dict[str, Any]):
            try:
                return ctor(**kwargs)
            except TypeError:
                # Fall back to positional endpoint list if supported
                if "endpoints" in kwargs:
                    return ctor(kwargs["endpoints"])
                raise

        try:
            ctor: Optional[Callable[..., Any]] = None
            if hasattr(kvclient, "KVClient"):
                ctor = getattr(kvclient, "KVClient")
            elif hasattr(kvclient, "Client"):
                ctor = getattr(kvclient, "Client")
            elif callable(getattr(kvclient, "connect", None)):
                ctor = getattr(kvclient, "connect")

            if ctor is None:
                raise ImportError("kvclient does not expose a usable client constructor")

            kwargs = {**self.client_kwargs}
            if self.endpoints:
                kwargs.setdefault("endpoints", self.endpoints)
            if self.namespace:
                kwargs.setdefault("namespace", self.namespace)
            if self.username:
                kwargs.setdefault("username", self.username)
            if self.password:
                kwargs.setdefault("password", self.password)

            client = _instantiate(ctor, kwargs)
            logger.info("YuanrongConnector initialized with endpoints=%s", kwargs.get("endpoints"))
            return client
        except Exception as exc:  # pragma: no cover - defensive initialization
            logger.error("Failed to initialize kvclient: %s", exc)
            raise

    def _store_value(self, key: str, value: bytes) -> None:
        if hasattr(self.client, "put"):
            self.client.put(key, value)
        elif hasattr(self.client, "set"):
            self.client.set(key, value)
        else:
            self.client[key] = value  # type: ignore[index]

    def _fetch_value(self, key: str) -> Any:
        if hasattr(self.client, "get"):
            return self.client.get(key)
        try:
            return self.client[key]  # type: ignore[index]
        except Exception:
            return None

    def _delete_value(self, key: str) -> None:
        if hasattr(self.client, "delete"):
            try:
                self.client.delete(key)
            except Exception:
                logger.debug("YuanrongConnector delete failed for %s", key)

    def put(
        self, from_stage: str, to_stage: str, request_id: str, data: Any
    ) -> tuple[bool, int, Optional[dict[str, Any]]]:
        try:
            serialized_data = self.serialize_obj(data)
            key = self._make_key(request_id, from_stage, to_stage)
            self._store_value(key, serialized_data)

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += len(serialized_data)
            logger.debug(
                "YuanrongConnector: stored %s (%s -> %s) %d bytes",
                key,
                from_stage,
                to_stage,
                len(serialized_data),
            )
            return True, len(serialized_data), None
        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error("YuanrongConnector put failed: %s", exc)
            return False, 0, None

    def get(
        self, from_stage: str, to_stage: str, request_id: str, metadata: Optional[dict[str, Any]] = None
    ) -> Optional[tuple[Any, int]]:
        retries = self.config.get("retries", 20)
        sleep_s = self.config.get("retry_interval", 0.05)
        key = self._make_key(request_id, from_stage, to_stage)

        for attempt in range(retries):
            try:
                raw_data = self._fetch_value(key)
                if raw_data:
                    data = self.deserialize_obj(raw_data)
                    payload_size = len(raw_data)
                    self._metrics["gets"] += 1
                    logger.debug(
                        "YuanrongConnector: retrieved %s (%s -> %s) %d bytes",
                        key,
                        from_stage,
                        to_stage,
                        payload_size,
                    )
                    return data, payload_size
            except Exception as exc:
                logger.debug("YuanrongConnector get attempt %s failed: %s", attempt, exc)

            if attempt < retries - 1:
                time.sleep(sleep_s)

        self._metrics["timeouts"] += 1
        logger.warning("YuanrongConnector: timeout waiting for %s", key)
        return None

    def cleanup(self, request_id: str) -> None:
        key_prefix = f"{request_id}/"
        # kvclient may or may not support iteration; attempt best-effort delete
        try:
            if hasattr(self.client, "scan"):
                for key in self.client.scan(prefix=key_prefix):
                    self._delete_value(key)
            else:
                # Attempt direct delete of common directional suffixes
                self._delete_value(key_prefix)
        except Exception:
            logger.debug("YuanrongConnector cleanup failed for %s", request_id)

    def health(self) -> dict[str, Any]:
        if kvclient is None:
            return {"status": "unhealthy", "error": "kvclient not available"}

        status = {"status": "healthy", "endpoints": self.endpoints}
        status.update(self._metrics)

        if hasattr(self.client, "ping"):
            try:
                status["ping"] = self.client.ping()
            except Exception as exc:
                status["status"] = "unhealthy"
                status["error"] = f"ping failed: {exc}"
        return status

    def close(self) -> None:
        if hasattr(self.client, "close"):
            try:
                self.client.close()
                logger.info("YuanrongConnector closed")
            except Exception as exc:
                logger.error("Error closing YuanrongConnector: %s", exc)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from __future__ import annotations

import time
from typing import Any, Optional

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

try:
    from datasystem.kv_client import KVClient, SetParam, WriteMode
except ImportError:
    logger.warning("Datasystem not available, YuanrongConnector will not work")
    KVClient = None


class YuanrongConnector(OmniConnectorBase):
    """Datasystem-based distributed connector for OmniConnector."""

    def __init__(self, config: dict[str, Any]):
        if KVClient is None:
            raise ImportError("Datasystem not available")

        self.config = config
        self.endpoints = config.get("endpoints") or []
        self.namespace = config.get("namespace")
        self.username = config.get("username")
        self.password = config.get("password")
        self.client_kwargs = config.get("client_kwargs", {})

        self.kv_client = self._init_client()

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
        """Instantiate and initialize a KVClient using the datasystem API."""

        try:
            kwargs = {**self.client_kwargs}
            if self.endpoints:
                kwargs.setdefault("endpoints", self.endpoints)
            if self.namespace:
                kwargs.setdefault("namespace", self.namespace)
            if self.username:
                kwargs.setdefault("username", self.username)
            if self.password:
                kwargs.setdefault("password", self.password)

            client = KVClient()
            client.init(**kwargs)
            logger.info("YuanrongConnector initialized with endpoints=%s", kwargs.get("endpoints"))
            return client
        except Exception as exc:  # pragma: no cover - defensive initialization
            logger.error("Failed to initialize kvclient: %s", exc)
            raise

    def _delete_value(self, key: str) -> None:
        if hasattr(self.kv_client, "delete"):
            try:
                self.kv_client.delete(key)
            except Exception:
                logger.debug("YuanrongConnector delete failed for %s", key)

    def put(
        self, from_stage: str, to_stage: str, request_id: str, data: Any
    ) -> tuple[bool, int, Optional[dict[str, Any]]]:
        try:
            serialized_data = self.serialize_obj(data)
            key = self._make_key(request_id, from_stage, to_stage)
            set_kwargs: dict[str, Any] = {"key": key, "value": serialized_data}
            if self.namespace:
                set_kwargs.setdefault("namespace", self.namespace)
            write_mode = self.config.get("write_mode")
            if write_mode:
                try:
                    set_kwargs["write_mode"] = (
                        write_mode
                        if isinstance(write_mode, WriteMode)
                        else WriteMode[write_mode]
                    )
                except Exception:
                    logger.debug("Invalid write_mode %s provided; ignoring", write_mode)
            set_param = SetParam(**set_kwargs)
            self.kv_client.set(set_param)

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
                get_kwargs: dict[str, Any] = {"key": key}
                if self.namespace:
                    get_kwargs.setdefault("namespace", self.namespace)
                raw_data = self.kv_client.get(**get_kwargs)
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
            if hasattr(self.kv_client, "scan"):
                for key in self.kv_client.scan(prefix=key_prefix):
                    self._delete_value(key)
            else:
                # Attempt direct delete of common directional suffixes
                self._delete_value(key_prefix)
        except Exception:
            logger.debug("YuanrongConnector cleanup failed for %s", request_id)

    def health(self) -> dict[str, Any]:
        if KVClient is None:
            return {"status": "unhealthy", "error": "kvclient not available"}

        status = {"status": "healthy", "endpoints": self.endpoints}
        status.update(self._metrics)

        if hasattr(self.kv_client, "ping"):
            try:
                status["ping"] = self.kv_client.ping()
            except Exception as exc:
                status["status"] = "unhealthy"
                status["error"] = f"ping failed: {exc}"
        return status

    def close(self) -> None:
        if hasattr(self.kv_client, "close"):
            try:
                self.kv_client.close()
                logger.info("YuanrongConnector closed")
            except Exception as exc:
                logger.error("Error closing YuanrongConnector: %s", exc)

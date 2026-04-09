# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import threading

import pytest
import vllm_omni.distributed.omni_connectors.connectors.yuanrong_transfer_engine_connector as yr_module

from vllm_omni.distributed.omni_connectors.connectors.yuanrong_transfer_engine_connector import (
    YuanrongTransferEngineConnector,
)
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeResult:
    def __init__(self, ok: bool = True, msg: str = "ok"):
        self._ok = ok
        self._msg = msg

    def is_ok(self) -> bool:
        return self._ok

    def is_error(self) -> bool:
        return not self._ok

    def get_msg(self) -> str:
        return self._msg

    def to_string(self) -> str:
        return self._msg


class FakeTransferEngine:
    _lock = threading.Lock()
    _regions: dict[str, list[tuple[int, int]]] = {}

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._regions = {}

    def __init__(self):
        self.endpoint = ""
        self._rpc_port = -1

    def initialize(self, local_hostname: str, protocol: str, device_name: str) -> FakeResult:
        self.endpoint = local_hostname
        self._rpc_port = int(local_hostname.rsplit(":", 1)[1])
        with self._lock:
            self._regions.setdefault(self.endpoint, [])
        return FakeResult()

    def get_rpc_port(self) -> int:
        return self._rpc_port

    def register_memory(self, buffer_addr: int, length: int) -> FakeResult:
        with self._lock:
            self._regions.setdefault(self.endpoint, []).append((buffer_addr, length))
        return FakeResult()

    def unregister_memory(self, buffer_addr: int) -> FakeResult:
        with self._lock:
            regions = self._regions.get(self.endpoint, [])
            self._regions[self.endpoint] = [region for region in regions if region[0] != buffer_addr]
        return FakeResult()

    def batch_transfer_sync_read(
        self,
        target_hostname: str,
        buffers: list[int],
        peer_buffer_addresses: list[int],
        lengths: list[int],
    ) -> FakeResult:
        with self._lock:
            remote_regions = list(self._regions.get(target_hostname, []))
            local_regions = list(self._regions.get(self.endpoint, []))

        for dst_addr, src_addr, length in zip(buffers, peer_buffer_addresses, lengths):
            if not self._contains(remote_regions, src_addr, length):
                return FakeResult(False, f"remote addr {src_addr} not registered")
            if not self._contains(local_regions, dst_addr, length):
                return FakeResult(False, f"local addr {dst_addr} not registered")
            ctypes.memmove(dst_addr, src_addr, length)

        return FakeResult()

    def finalize(self) -> FakeResult:
        return FakeResult()

    @staticmethod
    def _contains(regions: list[tuple[int, int]], addr: int, length: int) -> bool:
        for base, size in regions:
            if base <= addr and addr + length <= base + size:
                return True
        return False


@pytest.fixture(autouse=True)
def patch_transfer_engine(monkeypatch: pytest.MonkeyPatch):
    FakeTransferEngine.reset()
    monkeypatch.setattr(yr_module, "_load_transfer_engine_class", lambda *_args, **_kwargs: FakeTransferEngine)
    yield
    FakeTransferEngine.reset()


def _cfg(role: str) -> dict:
    return {
        "host": "127.0.0.1",
        "zmq_port": "auto",
        "rpc_port": "auto",
        "protocol": "ascend",
        "device_name": "npu:0",
        "memory_pool_size": 1024 * 1024,
        "memory_pool_device": "cpu",
        "role": role,
    }


def test_factory_creates_yuanrong_transfer_engine_connector():
    spec = ConnectorSpec(name="YuanrongTransferEngineConnector", extra=_cfg("sender"))
    connector = OmniConnectorFactory.create_connector(spec)
    try:
        assert isinstance(connector, YuanrongTransferEngineConnector)
        assert "YuanrongTransferEngineConnector" in OmniConnectorFactory.list_registered_connectors()
    finally:
        connector.close()


def test_get_round_trip_with_metadata_and_cleanup():
    sender = YuanrongTransferEngineConnector(_cfg("sender"))
    receiver = YuanrongTransferEngineConnector(_cfg("receiver"))

    try:
        payload = {"request_id": "req-1", "tokens": [1, 2, 3]}
        ok, size, metadata = sender.put("0", "1", "req-1", payload)
        assert ok
        assert metadata is not None

        result = receiver.get("0", "1", "req-1", metadata)
        assert result is not None
        received, recv_size = result
        assert received == payload
        assert recv_size == size

        key = YuanrongTransferEngineConnector._make_key("req-1", "0", "1")
        assert key not in sender._local_buffers
    finally:
        sender.close()
        receiver.close()


def test_get_without_metadata_queries_sender():
    sender = YuanrongTransferEngineConnector(_cfg("sender"))
    receiver = YuanrongTransferEngineConnector(_cfg("receiver"))

    try:
        payload = b"yuanrong-transfer" * 32
        ok, size, _ = sender.put("0", "1", "req-2", payload)
        assert ok

        receiver.update_sender_info(sender.host, sender.zmq_port)
        result = receiver.get("0", "1", "req-2", metadata=None)
        assert result is not None

        recv_buffer, recv_size = result
        try:
            assert recv_buffer.to_bytes() == payload
            assert recv_size == size
        finally:
            recv_buffer.release()

        key = YuanrongTransferEngineConnector._make_key("req-2", "0", "1")
        assert key not in sender._local_buffers
    finally:
        sender.close()
        receiver.close()

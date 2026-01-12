"""PoC callback sender with retry-until-stop and bounded buffer."""
import asyncio
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional

import aiohttp

from vllm.logger import init_logger
from .data import Artifact

logger = init_logger(__name__)

POC_CALLBACK_INTERVAL_SEC = float(os.environ.get("POC_CALLBACK_INTERVAL_SEC", "5"))
POC_CALLBACK_MAX_ARTIFACTS = int(os.environ.get("POC_CALLBACK_MAX_ARTIFACTS", "1000000"))
POC_CALLBACK_RETRY_BACKOFF_SEC = 1.0
POC_CALLBACK_RETRY_MAX_BACKOFF_SEC = 30.0


class CallbackSender:
    """Manages callback sending with retry and bounded buffer."""
    
    def __init__(
        self,
        callback_url: str,
        stop_event: asyncio.Event,
        k_dim: int = 12,
        max_artifacts: int = POC_CALLBACK_MAX_ARTIFACTS,
    ):
        self.callback_url = callback_url
        self.stop_event = stop_event
        self.k_dim = k_dim
        self.max_artifacts = max_artifacts
        
        self._buffer: deque[Artifact] = deque()
        self._metadata: Dict[str, Any] = {}
        self._pending_payload: Optional[Dict] = None
        self._task: Optional[asyncio.Task] = None
    
    def add_artifacts(self, artifacts: List[Artifact], metadata: Dict[str, Any]):
        """Add artifacts to buffer, dropping oldest if cap exceeded."""
        self._metadata = metadata
        for artifact in artifacts:
            self._buffer.append(artifact)
        
        while len(self._buffer) > self.max_artifacts:
            self._buffer.popleft()
    
    def clear(self):
        """Clear all buffered artifacts."""
        self._buffer.clear()
        self._pending_payload = None
    
    @property
    def buffered_count(self) -> int:
        return len(self._buffer)
    
    async def run(self):
        """Main sender loop - batches and sends with retry-until-stop."""
        last_send_time = time.time()
        backoff = POC_CALLBACK_RETRY_BACKOFF_SEC
        
        async with aiohttp.ClientSession() as session:
            while not self.stop_event.is_set():
                await asyncio.sleep(0.1)
                
                current_time = time.time()
                should_send = (
                    (self._buffer or self._pending_payload) and
                    (current_time - last_send_time >= POC_CALLBACK_INTERVAL_SEC)
                )
                
                if not should_send:
                    continue
                
                if self._pending_payload is None and self._buffer:
                    artifacts_to_send = list(self._buffer)
                    self._buffer.clear()
                    self._pending_payload = {
                        **self._metadata,
                        "artifacts": [{"nonce": a.nonce, "vector_b64": a.vector_b64} for a in artifacts_to_send],
                        "encoding": {"dtype": "f16", "k_dim": self.k_dim, "endian": "le"},
                    }
                
                if self._pending_payload:
                    success = await self._send_callback(session, self._pending_payload)
                    if success:
                        self._pending_payload = None
                        backoff = POC_CALLBACK_RETRY_BACKOFF_SEC
                        last_send_time = current_time
                    else:
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, POC_CALLBACK_RETRY_MAX_BACKOFF_SEC)
    
    async def _send_callback(self, session: aiohttp.ClientSession, payload: Dict) -> bool:
        """Send callback, return True on success."""
        try:
            async with session.post(
                f"{self.callback_url}/generated",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status < 400:
                    logger.debug(f"Callback sent: {len(payload.get('artifacts', []))} artifacts")
                    return True
                logger.warning(f"Callback HTTP {resp.status}")
                return False
        except Exception as e:
            logger.warning(f"Callback failed: {e}")
            return False


async def send_oneshot_callback(
    url: str,
    path: str,
    payload: Dict,
    stop_event: Optional[asyncio.Event] = None,
):
    """Send a single callback with retry-until-stop or success."""
    backoff = POC_CALLBACK_RETRY_BACKOFF_SEC
    
    async with aiohttp.ClientSession() as session:
        while True:
            if stop_event and stop_event.is_set():
                return False
            
            try:
                async with session.post(
                    f"{url}/{path}",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status < 400:
                        return True
                    logger.warning(f"Oneshot callback HTTP {resp.status}")
            except Exception as e:
                logger.warning(f"Oneshot callback failed: {e}")
            
            if stop_event is None:
                return False
            
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, POC_CALLBACK_RETRY_MAX_BACKOFF_SEC)

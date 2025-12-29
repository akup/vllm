import logging
from typing import List, Optional
from dataclasses import asdict

import aiohttp

from .data import ProofBatch, ValidatedBatch

logger = logging.getLogger(__name__)


class PoCCallbackSender:
    """Sends valid batches to callback URL with retry logic.
    
    Failed batches are stored and retried on subsequent send calls.
    """
    
    def __init__(self, callback_url: str, r_target: float, fraud_threshold: float):
        self.callback_url = callback_url.rstrip("/")
        self.r_target = r_target
        self.fraud_threshold = fraud_threshold
        self._session: Optional[aiohttp.ClientSession] = None
        self._generated_not_sent: List[ProofBatch] = []
        self._validated_not_sent: List[ValidatedBatch] = []
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def _post(self, endpoint: str, data: dict) -> bool:
        url = f"{self.callback_url}{endpoint}"
        try:
            session = await self._get_session()
            async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    logger.debug(f"Callback sent to {url}")
                    return True
                else:
                    logger.warning(f"Callback failed: {url} returned {resp.status}")
                    return False
        except Exception as e:
            logger.warning(f"Callback error for {url}: {e}")
            return False
    
    async def send_generated(self, batch: ProofBatch) -> None:
        """POST {callback_url}/generated with valid nonces (filtered by r_target).
        
        Retries previously failed batches.
        """
        valid_batch = batch.sub_batch(self.r_target)
        if len(valid_batch) > 0:
            self._generated_not_sent.append(valid_batch)
        
        failed = []
        for b in self._generated_not_sent:
            success = await self._post("/generated", asdict(b))
            if not success:
                failed.append(b)
        self._generated_not_sent = failed
    
    async def send_validated(self, batch: ValidatedBatch) -> None:
        """POST {callback_url}/validated with validation results.
        
        Retries previously failed batches.
        """
        self._validated_not_sent.append(batch)
        
        failed = []
        for b in self._validated_not_sent:
            success = await self._post("/validated", asdict(b))
            if not success:
                failed.append(b)
        self._validated_not_sent = failed
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    @property
    def pending_generated_count(self) -> int:
        return len(self._generated_not_sent)
    
    @property
    def pending_validated_count(self) -> int:
        return len(self._validated_not_sent)


from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

T = TypeVar("T")


def _is_retryable_http_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (408, 429, 500, 502, 503, 504)
    return False


@dataclass(frozen=True)
class RetryConfig:
    attempts: int = 6
    min_seconds: float = 1.0
    max_seconds: float = 30.0


def async_retry(cfg: RetryConfig) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    def _decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        return retry(
            reraise=True,
            retry=retry_if_exception(_is_retryable_http_error),
            stop=stop_after_attempt(cfg.attempts),
            wait=wait_exponential_jitter(
                initial=cfg.min_seconds,
                max=cfg.max_seconds,
                jitter=1.0 + random.random(),
            ),
        )(fn)

    return _decorator



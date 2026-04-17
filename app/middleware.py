import logging
import uuid
from contextvars import ContextVar

from fastapi import Request

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    return _request_id.get()


class RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()  # type: ignore[attr-defined]
        return True


async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    token = _request_id.set(rid)
    try:
        response = await call_next(request)
    finally:
        _request_id.reset(token)
    response.headers["X-Request-ID"] = rid
    return response

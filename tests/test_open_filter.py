import sys
from pathlib import Path
from typing import Dict

import pytest
from starlette.requests import Request

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from email_marketing.tracking.server import _should_count_open


def make_request(headers: Dict[str, str]) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/pixel",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("1.2.3.4", 1234),
        "query_string": b"",
    }
    return Request(scope)


def test_allows_normal_user_agent() -> None:
    req = make_request({"User-Agent": "Mozilla", "X-Forwarded-For": "1.2.3.4"})
    assert _should_count_open(req)


def test_blocks_gmail_prefetch_without_xff() -> None:
    req = make_request({"User-Agent": "GoogleImageProxy"})
    assert _should_count_open(req) is False


def test_allows_gmail_with_xff() -> None:
    req = make_request({"User-Agent": "GoogleImageProxy", "X-Forwarded-For": "9.8.7.6"})
    assert _should_count_open(req)


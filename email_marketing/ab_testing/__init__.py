"""A/B testing utilities.

Functions in this module assign recipients to A/B variants for experiments.
The current implementation uses a deterministic hash of the recipient's
address so that the same address always receives the same variant.  Future
enhancements could incorporate weighted distributions or multi‑armed
bandits.
"""

from __future__ import annotations

import hashlib
from typing import Literal

Variant = Literal["A", "B"]



def assign_variant(recipient: str) -> Variant:
    """Assign a recipient to variant A or B deterministically.

    Args:
        recipient: The recipient's email address.

    Returns:
        "A" or "B" depending on the hash of the address.
    """
    digest = hashlib.sha256(recipient.encode("utf-8")).digest()
    # Use the least significant bit to assign a variant
    return "A" if digest[0] % 2 == 0 else "B"


__all__ = ["assign_variant", "Variant"]

"""Source connectors — local-only ingestion (folders, archives, PDFs).

Remote scraping has been removed; user-facing tools (Discord bot,
browser extension) handle download and upload bnl/folder via the API.
"""

from .constants import IMAGE_EXTS
from .local import LocalSource

__all__ = ["LocalSource", "IMAGE_EXTS"]

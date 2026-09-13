"""Anonymous workspace capabilities. Tokens stay in the browser; only hashes persist."""
from contextvars import ContextVar

# None is reserved for explicit local maintenance, outside HTTP requests.
workspace = ContextVar("visionlog_workspace", default=None)

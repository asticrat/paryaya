"""Shared mutable state for the API process (populated during lifespan startup)."""
import time

MODULE_STATE: dict = {}
_START_TIME: float = time.time()

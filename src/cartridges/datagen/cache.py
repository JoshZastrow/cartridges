"""Compatibility wrapper for the old datagen cache module."""

from cartridges.storage.cache import get_cached, hash_image_bytes, init_cache, put_cached

__all__ = ["get_cached", "hash_image_bytes", "init_cache", "put_cached"]

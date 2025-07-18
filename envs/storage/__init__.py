from .cache import CacheBase
from .persist import PersistBase

__all__ = ['CacheBase', 'PersistBase']

CACHE_REGISTRY = {
    'cache': CacheBase,
    'persist': PersistBase
}


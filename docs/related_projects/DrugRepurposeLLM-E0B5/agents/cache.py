import json
from pathlib import Path
from typing import Any, Dict, Optional


class FileCache:
    """A file-based caching system that stores data in JSON format.

    Args:
        cache_dir (str): Directory path where cache files will be stored. Defaults to ".cache".
    """

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Convert key to a filename-safe format and return the full cache path.

        Args:
            key (str): Cache key to be converted

        Returns:
            Path: Path object pointing to the cache file location
        """
        safe_key = "".join(c if c.isalnum() else "_" for c in key)
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> Optional[Dict[Any, Any]]:
        """Retrieve cached data for the given key.

        Args:
            key (str): Cache key to retrieve

        Returns:
            Optional[Dict[Any, Any]]: Cached data if exists, None otherwise
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def set(self, key: str, value: Dict):
        """Store data in cache for the given key.

        Args:
            key (str): Cache key to store
            value (Dict): Data to be cached
        """
        cache_path = self._get_cache_path(key)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)

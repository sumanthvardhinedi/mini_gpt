"""
Phase 12: Local Document Loader for MiniGPT RAG.

Supports loading `.txt` and `.md` documents from individual file paths or directories.
Each loaded document is normalized into:
```python
{
    "text": "...",
    "source": "filename.md",
    "metadata": {
        "path": "/abs/or/rel/path/filename.md",
        "extension": ".md",
        "char_count": 1234,
    },
}
```
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_document(
    file_path: Union[str, Path],
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load a single local `.txt` or `.md` file into a structured document dictionary.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file path, got directory: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension {ext!r} for {path.name}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    text = path.read_text(encoding="utf-8")
    metadata: Dict[str, Any] = {
        "path": str(path),
        "extension": ext,
        "char_count": len(text),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "text": text,
        "source": path.name,
        "metadata": metadata,
    }


def load_documents(
    source_input: Union[str, Path, Sequence[Union[str, Path]]],
) -> List[Dict[str, Any]]:
    """
    Load multiple `.txt` and `.md` documents from either:
    - a directory path (sorted deterministically by filename),
    - a single file path, or
    - a sequence of file paths.
    """
    if isinstance(source_input, (str, Path)):
        target = Path(source_input)
        if not target.exists():
            raise FileNotFoundError(f"Document path does not exist: {target}")
        if target.is_file():
            return [load_document(target)]
        files = sorted(
            p
            for p in target.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        return [load_document(p) for p in files]

    if isinstance(source_input, Sequence):
        docs: List[Dict[str, Any]] = []
        for item in source_input:
            docs.append(load_document(item))
        return docs

    raise TypeError(
        "source_input must be a directory path, file path, or sequence of file paths."
    )

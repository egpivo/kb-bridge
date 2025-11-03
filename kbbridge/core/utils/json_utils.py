import json
import re
from typing import Any, Dict, List, Union

UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}"
)

_OBJ_RE = re.compile(r"\{[^{}]*\}")


def parse_json_from_markdown(json_string: str) -> dict:
    """Parse JSON from markdown code blocks or plain text

    Args:
        json_string: String potentially containing JSON in markdown code blocks

    Returns:
        Dict with "result" key containing the parsed array

    Raises:
        ValueError: If no valid JSON array is found
    """
    match = re.search(r"```json\s*(\[\s*[\s\S]*?\])\s*```", json_string, re.IGNORECASE)
    if not match:
        match = re.search(r"```\s*(\[\s*[\s\S]*?\])\s*```", json_string)
    if not match:
        raise ValueError("No JSON array found in the provided string.")

    json_block = match.group(1)
    result_array = json.loads(json_block)

    # Optionally verify that we indeed extracted a list of keyword sets
    if not isinstance(result_array, list) or not all(
        isinstance(x, list) and all(isinstance(item, str) for item in x)
        for x in result_array
    ):
        raise ValueError(
            "Extracted JSON is not an array of keyword sets (arrays of strings)."
        )

    return {"result": result_array}


def parse_dataset_ids(raw: str) -> List[str]:
    """
    Flexibly parse a possibly-quoted, nested JSON array-of-strings into a Python list.
    If all else fails, extract UUIDs by regex from the raw input.

    Args:
        raw: Raw string containing dataset IDs

    Returns:
        List of dataset ID strings
    """
    s: Union[str, Any] = raw.strip()

    # 0) If there are UUIDs anywhere in the raw, return them immediately.
    #    This handles the really messy quoting cases.
    uuids = UUID_PATTERN.findall(raw)
    if uuids:
        return uuids

    # 1) Unwrap repeated JSON string layers
    while True:
        if (
            isinstance(s, str)
            and len(s) >= 2
            and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"))
        ):
            try:
                s = json.loads(s)
                continue
            except json.JSONDecodeError:
                break
        break

    if isinstance(s, list):
        return [str(x) for x in s if isinstance(x, (str, int, float))]

    if isinstance(s, str):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x) for x in arr if isinstance(x, (str, int, float))]
        except json.JSONDecodeError:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return parts

    return []


def parse_dataset_info(raw: str) -> List[Dict[str, Any]]:
    """
    Generic parser for dataset information that can handle:
    - Simple arrays of strings/IDs: ["id1", "id2", ...]
    - Complex arrays of dictionaries: [{"id": "id1"}, {"folder": "folder1"}, ...]
    - Mixed nested JSON string layers
    - Fallback UUID extraction for malformed inputs

    Args:
        raw: Raw string containing dataset information

    Returns:
        List of dictionaries with "id" and "source_path" keys
    """
    s: Union[str, Any] = raw.strip()

    # 0) If there are UUIDs anywhere in the raw, return them as id objects
    uuids = UUID_PATTERN.findall(raw)
    if uuids and not _looks_like_structured_json(raw):
        return [{"id": uuid, "source_path": ""} for uuid in uuids]

    # 1) Unwrap repeated JSON string layers
    while True:
        if (
            isinstance(s, str)
            and len(s) >= 2
            and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"))
        ):
            try:
                s = json.loads(s)
                continue
            except json.JSONDecodeError:
                break
        break

    # 2) If we now have a real list, process it
    if isinstance(s, list):
        return _process_list_items(s)

    # 3) If it's still a string, try JSON-loading as array
    if isinstance(s, str):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return _process_list_items(arr)
        except json.JSONDecodeError:
            # Salvage dictionaries from malformed JSON text
            salvaged = _salvage_id_folder_pairs(s)
            if salvaged:
                return salvaged

            # fallback: split on commas and treat as simple IDs (folder empty)
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return [{"id": part, "source_path": ""} for part in parts]

    # 4) Nothing workable found
    return []


def _looks_like_structured_json(raw: str) -> bool:
    """Check if the raw string looks like it contains structured JSON objects."""
    return "{" in raw and "}" in raw


def _process_list_items(items: List[Any]) -> List[Dict[str, Any]]:
    """Return a sanitized list with exactly `id` and `source_path` keys.

    • Only items that are dictionaries **and** contain an ``id`` key are kept.
    • The ``source_path`` key is always present – empty string if missing.
    • All other keys are discarded.
    """

    result: List[Dict[str, Any]] = []

    for item in items:
        if isinstance(item, dict):
            id_value = item.get("id")
            if id_value in (None, ""):
                continue

            source_path_value = item.get("source_path", "")

            result.append(
                {
                    "id": str(id_value),
                    "source_path": str(source_path_value)
                    if source_path_value is not None
                    else "",
                }
            )
        elif isinstance(item, (str, int, float)):
            # Scalar treated as id-only, source_path empty
            result.append({"id": str(item), "source_path": ""})
        # Other types ignored
    return result


def _salvage_id_folder_pairs(raw: str) -> List[Dict[str, Any]]:
    """Best-effort extraction of ``{"id": …, "source_path": …}`` pairs from *raw*.

    This is used when the input is *not* valid JSON but still contains
    recognisable object fragments.  Only objects with an ``id`` field are kept.
    Any missing ``source_path`` becomes an empty string.  The function is deliberately
    permissive and *never* raises – it either returns a non-empty list or ``[]``.
    """

    out: List[Dict[str, Any]] = []

    for m in _OBJ_RE.finditer(raw):
        snippet = m.group(0)

        try:
            obj = json.loads(snippet)
        except Exception:
            obj = None

        if isinstance(obj, dict) and "id" in obj:
            out.append(
                {
                    "id": str(obj["id"]),
                    "source_path": str(obj.get("source_path", "")),
                }
            )
            continue

        id_match = re.search(r"\"id\"\s*:\s*([^,}]+)", snippet)
        if not id_match:
            continue

        id_token = id_match.group(1).strip()
        if id_token and id_token[0] in "'\"" and id_token[-1] == id_token[0]:
            id_token = id_token[1:-1]

        folder_match = re.search(r"\"source_path\"\s*:\s*\"([^\"]*)\"", snippet)
        folder_val = folder_match.group(1) if folder_match else ""

        out.append({"id": id_token, "source_path": folder_val})

    return out

import json
import re
from typing import Any, List, Union

UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}"
)


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

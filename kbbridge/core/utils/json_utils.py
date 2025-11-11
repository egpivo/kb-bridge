import json
import re

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

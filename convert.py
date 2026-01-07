import json
import argparse
from typing import Any, Dict, List, Optional


def make_id(record: Dict[str, Any], fallback_index: int) -> str:
    """
    Build an ID similar in spirit to your example.

    Priority:
    1. uuid + "_" + idx  (if both exist)
    2. uuid              (if exists)
    3. "sample_<idx>"    (if idx exists)
    4. "sample_<fallback_index>"
    """
    uuid: Optional[str] = record.get("uuid")
    idx = record.get("idx")

    if uuid is not None and idx is not None:
        return f"{uuid}_{idx}"
    if uuid is not None:
        return str(uuid)
    if idx is not None:
        return f"sample_{idx}"
    return f"sample_{fallback_index}"


def convert_record(record: Dict[str, Any], fallback_index: int) -> Dict[str, Any]:
    """
    Convert one input JSONL record to the desired format:

    {
      "id": "...",
      "conversations": [
        { "from": "human", "value": "<prompt>" },
        { "from": "gpt",   "value": "<response>" }
      ]
    }
    """
    # Handle both 'prompt' and 'instruction' fields
    prompt = record.get("prompt") or record.get("instruction", "")
    response = record.get("response", "")
    reasoning = record.get("reasoning_content", "")

    return {
        "id": make_id(record, fallback_index),
        "conversations": [
            {
                "from": "human",
                "value": prompt,
            },
            {
                "from": "gpt",
                "thinking": reasoning,
                "content": response,
            },

        ],
    }


def convert_jsonl_to_array(input_path: str, output_path: str) -> None:
    """
    Read an input JSONL file and write a single JSON file containing an array
    of objects in the desired format.
    """
    output_data: List[Dict[str, Any]] = []

    with open(input_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            converted = convert_record(record, fallback_index=i)
            output_data.append(converted)

    # Write as one JSON array, pretty-printed
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(output_data, fout, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL (prompt/response) to conversation array format."
    )
    parser.add_argument("input", help="Path to input JSONL file")
    parser.add_argument("output", help="Path to output JSON file")
    args = parser.parse_args()
    print(f"Converting {args.input} to {args.output}...")
    convert_jsonl_to_array(args.input, args.output)


if __name__ == "__main__":
    main()

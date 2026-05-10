#!/bin/bash
#
# Deduplicate a JSONL file by the "id" field, keeping the last occurrence.
#
# Usage:
#   ./dedup_jsonl.sh input.jsonl output.jsonl
#

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.jsonl> [output.jsonl]"
    echo "  If output is omitted, the input file is deduped in place."
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-$INPUT}"

if [ ! -f "$INPUT" ]; then
    echo "Error: $INPUT not found"
    exit 1
fi

BEFORE=$(wc -l < "$INPUT" | tr -d ' ')
TMPFILE=$(mktemp "${INPUT}.dedup.XXXXXX")

tac "$INPUT" | python3 -c "
import json,sys
seen=set()
lines=[]
for l in sys.stdin:
    i=json.loads(l).get('id')
    if i not in seen:
        seen.add(i)
        lines.append(l)
for l in reversed(lines):
    sys.stdout.write(l)
" > "$TMPFILE"

AFTER=$(wc -l < "$TMPFILE" | tr -d ' ')
REMOVED=$((BEFORE - AFTER))

mv "$TMPFILE" "$OUTPUT"

echo "Before: $BEFORE"
echo "After:  $AFTER"
echo "Removed: $REMOVED duplicates"

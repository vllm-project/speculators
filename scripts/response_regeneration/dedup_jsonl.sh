#!/bin/bash
#
# Deduplicate a JSONL file by the "id" field, keeping the last occurrence.
#
# Usage:
#   ./dedup_jsonl.sh input.jsonl output.jsonl
#

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input.jsonl> <output.jsonl>"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"

if [ ! -f "$INPUT" ]; then
    echo "Error: $INPUT not found"
    exit 1
fi

BEFORE=$(wc -l < "$INPUT" | tr -d ' ')

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
" > "$OUTPUT"

AFTER=$(wc -l < "$OUTPUT" | tr -d ' ')
REMOVED=$((BEFORE - AFTER))

echo "Before: $BEFORE"
echo "After:  $AFTER"
echo "Removed: $REMOVED duplicates"

#!/usr/bin/env bash

set -e

URL="${URL:-http://127.0.0.1:8000/v1/classify}"
CONCURRENCY="${CONCURRENCY:-16}"
REQUESTS="${REQUESTS:-100}"

echo "Bayan API Load Test"
echo "URL: $URL"
echo "Concurrency: $CONCURRENCY"
echo "Requests: $REQUESTS"

BODY='{"text":"الشارع مظلم ونحتاج إصلاح الإنارة"}'

if command -v hey >/dev/null 2>&1; then
    hey \
        -n "$REQUESTS" \
        -c "$CONCURRENCY" \
        -m POST \
        -H "Content-Type: application/json" \
        -d "$BODY" \
        "$URL"
else
    echo "The 'hey' load-testing tool is not installed."
    echo "Install it and rerun this script."
    exit 1
fi

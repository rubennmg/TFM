#!/bin/bash
# Execute benchmarks for all pipelines defined in the pipelines directory.

PIPELINES_DIR="pipelines"
PROFILES=()

for pipeline_file in "$PIPELINES_DIR"/*.json; do
    if [ -f "$pipeline_file" ]; then
        profile_name=$(basename "$pipeline_file" .json)
        PROFILES+=("$profile_name")
    fi
done

if [ ${#PROFILES[@]} -eq 0 ]; then
    echo "Error: No pipelines found in $PIPELINES_DIR"
    exit 1
fi

echo "Found ${#PROFILES[@]} pipelines:"
for profile in "${PROFILES[@]}"; do
    echo "  - $profile"
done

echo ""
echo "Executing benchmarks..."
echo ""

for profile in "${PROFILES[@]}"; do
    echo "================================"
    echo "Executing benchmark: $profile"
    echo "================================"
    pytest benchmark/run.py --bench-profile="$PIPELINES_DIR/$profile.json"
    if [ $? -ne 0 ]; then
        echo "The benchmark '$profile' finished with an error"
    else
        echo "Benchmark '$profile' completed successfully"
    fi
    echo ""
done

echo "================================"
echo "All benchmarks have finished"
echo "================================"

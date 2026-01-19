#!/bin/bash

INPUT_DIR="input_files"
SCRIPT_NAME="claude-2-mul-thread-gpt5-vol-selection-output.py"
MAX_POPULATION=2000

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory $INPUT_DIR does not exist"
    exit 1
fi

# Get all CSV files, sort by startdate (extracted from filename)
mapfile -t files < <(ls -1 "$INPUT_DIR"/*.csv 2>/dev/null | sort)

if [ ${#files[@]} -eq 0 ]; then
    echo "Error: No CSV files found in $INPUT_DIR"
    exit 1
fi

# Process each file
for filepath in "${files[@]}"; do
    filename=$(basename "$filepath")
    
    # Generate output and log filenames
    output_file="${filename%.csv}.out.csv"
    log_file="${filename%.csv}.log"
    
    echo "Processing: $filename"

    echo "output_file: $output_file"
    echo "log_file: $log_file"

    # Run the Python script
    python "$SCRIPT_NAME" \
        --max-population="$MAX_POPULATION" \
        --log-file="$log_file" \
        --input-file="$filename" \
        --output-file="$output_file"    
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: $filename"
    else
        echo "✗ Failed: $filename"
    fi
done

echo "All files processed"

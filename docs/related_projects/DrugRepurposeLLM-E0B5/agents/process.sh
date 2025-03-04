#!/bin/bash

echo "Starting process at $(date)" >> output.txt
echo "----------------------------------------" >> output.txt

for i in {6..84}
do
    file="new_${i}.csv"
    if [ -f "$file" ]; then
        echo "File name: $file" >> output.txt
        echo "Processing $file..." >> output.txt
        python integrate_agent.py --csv "$file" >> output.txt 2>&1
        echo "Completed processing: $file" >> output.txt
        echo "----------------------------------------" >> output.txt
    else
        echo "Warning: File $file not found" >> output.txt
        echo "----------------------------------------" >> output.txt
    fi
done

echo "Process completed at $(date)" >> output.txt

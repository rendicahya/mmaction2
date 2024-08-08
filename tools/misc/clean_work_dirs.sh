#!/bin/bash

# Initialize counters for deleted files and freed space
deleted_files=0
freed_space=0

script_dir="$(dirname "$0")"

# Change to the base directory containing the work_dirs
cd "$script_dir/../.." || exit

# Iterate over all directories starting with "work_dirs"
for dir in work_dirs*; do
    # Check if it is indeed a directory
    if [[ -d "$dir" ]]; then
        # Change to the specific work_dirs directory
        cd "$dir" || continue

        # Find and remove last_checkpoint files and their target files
        find . -type f -name "last_checkpoint" | while read -r checkpoint_file; do
            checkpoint_path=$(head -n 1 "$checkpoint_file" | tr -d '\n\r')

            # Check if the target checkpoint file exists and remove it
            if [[ -f "$checkpoint_path" ]]; then
                file_size=$(stat -c %s "$checkpoint_path") # Get file size
                rm "$checkpoint_path"
                deleted_files=$((deleted_files + 1))
                freed_space=$((freed_space + file_size))
            fi

            # Remove the last_checkpoint file
            file_size=$(stat -c %s "$checkpoint_file") # Get file size
            rm "$checkpoint_file"
            deleted_files=$((deleted_files + 1))
            freed_space=$((freed_space + file_size))
        done

        # Return to the base directory
        cd - || exit
    fi
done

# Convert freed space to human-readable format
freed_space_human=$(numfmt --to=iec-i --suffix=B "$freed_space")

# Display the summary
echo "Summary:"
echo "Total deleted files: $deleted_files"
echo "Total freed space: $freed_space_human"

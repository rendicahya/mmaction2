#!/bin/bash

deleted_files=0
freed_space=0

script_dir="$(dirname "$0")"

cd "$script_dir/../.." || exit

for dir in work_dirs*; do
    if [[ -d "$dir" ]]; then
        cd "$dir" || continue

        while IFS= read -r checkpoint_file; do
            checkpoint_path=$(head -n 1 "$checkpoint_file" | tr -d '\n\r')

            if [[ -f "$checkpoint_path" ]]; then
                file_size=$(stat -c %s "$checkpoint_path") # Get file size
                rm "$checkpoint_path"
                deleted_files=$((deleted_files + 1))
                freed_space=$((freed_space + file_size))
            fi

            file_size=$(stat -c %s "$checkpoint_file") # Get file size
            rm "$checkpoint_file"
            deleted_files=$((deleted_files + 1))
            freed_space=$((freed_space + file_size))
        done < <(find . -type f -name "last_checkpoint")

        cd ~- || exit
    fi
done

freed_space_human=$(numfmt --to=iec-i --suffix=B "$freed_space")

echo "Summary:"
echo "Total deleted files: $deleted_files"
echo "Total freed space: $freed_space_human"

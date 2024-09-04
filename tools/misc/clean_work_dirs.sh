#!/bin/bash

deleted_files=0
freed_space=0
files_to_remove=false

# Check if "-y" option is passed
delete_files=false
if [[ "$1" == "-y" ]]; then
    delete_files=true
fi

script_dir="$(dirname "$0")"

# Navigate to the parent directory
cd "$script_dir/../.." || exit

# Iterate over each directory that starts with "work_dirs"
for dir in work_dirs*; do
    if [[ -d "$dir" ]]; then
        cd "$dir" || continue

        # Find all "train" directories and check for sibling "test" directory
        while IFS= read -r train_dir; do
            test_dir="$(dirname "$train_dir")/test"
            if [[ -d "$test_dir" ]]; then
                # If "test" directory exists, delete or simulate deletion of .pth files
                while IFS= read -r pth_file; do
                    if [[ -f "$pth_file" ]]; then
                        file_size=$(stat -c %s "$pth_file")
                        file_size_formatted=$(numfmt --to=iec-i --suffix=B "$file_size")

                        if [[ "$delete_files" == true ]]; then
                            rm "$pth_file"
                            echo "Deleted: $pth_file ($file_size_formatted)"
                        else
                            echo "Simulating deletion: $pth_file ($file_size_formatted)"
                            files_to_remove=true
                        fi
                        deleted_files=$((deleted_files + 1))
                        freed_space=$((freed_space + file_size))
                    fi
                done < <(find "$train_dir" -type f -name "*.pth")
            else
                # Report "train" directories without a sibling "test" directory
                echo "Warning: 'train' directory '$train_dir' does not have a sibling 'test' directory."
            fi
        done < <(find . -type d -name "train")

        # Delete or simulate deletion of files listed in "last_checkpoint"
        while IFS= read -r checkpoint_file; do
            checkpoint_path=$(head -n 1 "$checkpoint_file" | tr -d '\n\r')

            if [[ -f "$checkpoint_path" ]]; then
                file_size=$(stat -c %s "$checkpoint_path")
                file_size_formatted=$(numfmt --to=iec-i --suffix=B "$file_size")

                if [[ "$delete_files" == true ]]; then
                    rm "$checkpoint_path"
                    echo "Deleted: $checkpoint_path ($file_size_formatted)"
                else
                    echo "Simulating deletion: $checkpoint_path ($file_size_formatted)"
                    files_to_remove=true
                fi
                deleted_files=$((deleted_files + 1))
                freed_space=$((freed_space + file_size))
            fi

            file_size=$(stat -c %s "$checkpoint_file")
            file_size_formatted=$(numfmt --to=iec-i --suffix=B "$file_size")

            if [[ "$delete_files" == true ]]; then
                rm "$checkpoint_file"
                echo "Deleted: $checkpoint_file ($file_size_formatted)"
            else
                echo "Simulating deletion: $checkpoint_file ($file_size_formatted)"
                files_to_remove=true
            fi
            deleted_files=$((deleted_files + 1))
            freed_space=$((freed_space + file_size))
        done < <(find . -type f -name "last_checkpoint")

        # Return to the previous directory
        cd ~- || exit
    fi
done

# Convert total freed space to a human-readable format in binary units
freed_space_formatted=$(numfmt --to=iec-i --suffix=B "$freed_space")

# Print results
if [[ "$delete_files" == true ]]; then
    echo "Total deleted files: $deleted_files"
    echo "Total freed space: $freed_space_formatted"
elif [[ "$files_to_remove" == true ]]; then
    echo "Simulated deleted files: $deleted_files"
    echo "Simulated freed space: $freed_space_formatted"
    echo "Add '-y' option to actually remove files."
fi

#!/bin/bash

script_dir="$(dirname "$0")"

cd "$script_dir/../../work_dirs" || exit

find . -type f -name "last_checkpoint" | while read -r checkpoint_file; do
    checkpoint_path=$(head -n 1 "$checkpoint_file" | tr -d '\n\r')
    
    if [[ -f "$checkpoint_path" ]]; then
        rm "$checkpoint_path"
    fi

    rm "$checkpoint_file"
done

cd - || exit

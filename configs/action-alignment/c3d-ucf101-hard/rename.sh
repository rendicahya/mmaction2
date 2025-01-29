#!/bin/bash

for file in *; do
  new_filename="${file/-p0.5/}"
  if [[ "$new_filename" != "$file" ]]; then
    mv "$file" "$new_filename"
  fi
done
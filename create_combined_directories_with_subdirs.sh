#!/bin/bash

# Define the first part of the outer directory names
part1=("l" "p" "i" "t950" "t300" "t50")

# Define the second part of the outer directory names
part2=("nat" "ast" "mat" "psy" "mus")

# Define the inner directories
inner_dirs=("decoder" "last_token_pdf" "word2vec")

# Define the subdirectories for 'decoder' and 'last_token_pdf'
sub_dirs=("decoder_{1..12}")

# Loop through each combination of part1 and part2
for p1 in "${part1[@]}"; do
  for p2 in "${part2[@]}"; do
    outer_dir="output_${p1}_${p2}"
    
    # Create the outer directory
    mkdir -p "$outer_dir"
    
    # Create the inner directories inside the outer directory
    for dir in "${inner_dirs[@]}"; do
      mkdir -p "$outer_dir/$dir"
      
      # If the directory is 'decoder' or 'last_token_pdf', create subdirectories
      if [[ "$dir" == "decoder" || "$dir" == "last_token_pdf" ]]; then
        for i in {1..12}; do
          mkdir -p "$outer_dir/$dir/decoder_$i"
        done
      fi
    done
    
    echo "Created directory structure for $outer_dir"
  done
done

# Print a success message
echo "All directory structures created successfully."

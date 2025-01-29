import os

def rename_files(base_path):
    for root, dirs, files in os.walk(base_path):
        for file_name in files:
            new_name = file_name
            if '-v' in new_name:
                new_name = new_name.replace('-v', '-a')
            if '-m' in new_name:
                new_name = new_name.replace('-m', '-s')
            if new_name != file_name:  # Only rename if changes are made
                old_path = os.path.join(root, file_name)
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                print(f'Renamed: {old_path} -> {new_path}')

# Specify the base path (current directory)
base_path = os.getcwd()

# Call the function
rename_files(base_path)

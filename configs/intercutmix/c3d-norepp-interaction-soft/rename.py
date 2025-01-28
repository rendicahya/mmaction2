from pathlib import Path

directory = Path(".")

for file in directory.glob("**/*"):
    if "interaction" in file.name:
        new_name = file.name.replace("interaction", "interaction")
        new_path = file.with_name(new_name)
        
        file.rename(new_path)
        print(f"Renamed: {file} -> {new_path}")

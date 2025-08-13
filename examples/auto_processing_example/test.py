import zipfile
import time

zip_filename = "test_files.zip"

with zipfile.ZipFile(zip_filename, mode="w", compression=zipfile.ZIP_STORED) as zf:
    for i in range(1, 101):
        file_name = f"file_{i:03d}.txt"
        file_content = f"This is file number {i}\n"
        zf.writestr(file_name, file_content)
        time.sleep(0.05)  # Simulate some delay in file creation

print(f"Created {zip_filename} with 100 text files (store mode).")
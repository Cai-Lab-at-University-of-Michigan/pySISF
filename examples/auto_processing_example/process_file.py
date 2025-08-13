#!/usr/bin/env python3
"""
Example file processor script for use with folder_watchdog.py
This script demonstrates how to process files detected by the watchdog.
"""

import sys
import os
import time
from pathlib import Path

def main():
    """Main function to handle command line arguments and process files."""
    if len(sys.argv) != 2:
        print("Usage: python process_file.py <file_path>")
        print("This script is designed to be called by folder_watchdog.py")
        sys.exit(1)

    file_path = sys.argv[1]

    print(f"File processor started for: {file_path}")
    print(f"Process ID: {os.getpid()}")

    # Process the file
    #process_file(file_path)

    print("File processing completed")

    os.remove(file_path)


if __name__ == "__main__":
    main()

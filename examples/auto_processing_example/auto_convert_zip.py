import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from concurrent.futures import ThreadPoolExecutor
import zipfile


class FileChangeHandler(FileSystemEventHandler):
    """Handles file system events and manages subprocess execution."""

    def __init__(self, script_path, max_workers=4):
        """
        Initialize the file change handler.

        Args:
            script_path (str): Path to the Python script to run for each file change
            max_workers (int): Maximum number of concurrent subprocesses (default: 4)
        """
        self.script_path = script_path
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running_processes = {}  # Track running processes by file path
        self.lock = threading.Lock()

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self.process_file(event.src_path, "modified")

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self.process_file(event.src_path, "created")

    def on_moved(self, event):
        """Handle file move/rename events."""
        if not event.is_directory:
            # Cancel any existing process for the old path
            self.cancel_process(event.src_path)
            # Start new process for the new path
            self.process_file(event.dest_path, "moved")

    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory:
            # Cancel any running process for the deleted file
            self.cancel_process(event.src_path)

    def process_file(self, file_path, event_type):
        """
        Process a file by running the specified script as a subprocess.

        Args:
            file_path (str): Path to the file that changed
            event_type (str): Type of event (created, modified, moved)
        """

        # Skip temporary files and hidden files
        skip_patterns = [
            '.',  # Hidden files
            '~',  # Temporary files
            '.tmp',  # Temporary files
            '.log',  # Log files
            '.lock',  # Lock files
            '__pycache__',  # Python cache
            '.DS_Store',  # macOS system files
        ]
        if  any(file_path.startswith(pattern) or file_path.endswith(pattern) for pattern in skip_patterns) or '.zip' not in file_path:
            return

        with self.lock:
            # Cancel any existing process for this file
            if file_path in self.running_processes:
                print(f"Skip because already running for {file_path}")
                return
            #print(f"Cancelling existing process for {file_path}")
                #self.running_processes[file_path].cancel()

            # Submit new task to executor
            future = self.executor.submit(self.run_script, file_path, event_type)
            self.running_processes[file_path] = future

            # Add callback to clean up completed processes
            #future.add_done_callback(lambda f: self.cleanup_process(file_path, f))

    def run_script(self, file_path, event_type):
        """
        Run the specified Python script with the file path as an argument.

        Args:
            file_path (str): Path to the file that changed
            event_type (str): Type of event that triggered this
        """
        print(f"Processing {event_type} file: {file_path}")
        process = None

        try:
            if not zipfile.is_zipfile(file_path):
                raise ValueError(f"not a valid zip file.")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Check the namelist because that tells if the zip is corrupt
                zip_ref.namelist()
        except Exception as e:
            print(f"Skipping processing {file_path} because it failed the zip check")
            self.cleanup_process(file_path)
            return

        try:
            # Run the script with the file path as an argument
            cmd = [sys.executable, self.script_path, file_path]

            # Start the subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, #subprocess.PIPE,
                text=True
            )

            # Wait for completion and capture output
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                print(f"✓ Successfully processed {file_path}")
            else:
                print(f"✗ Error processing {file_path} (exit code: {process.returncode})")

            if stdout.strip():
                print(f"STDOUT:\n", stdout.strip())
            if stderr.strip():
                print(f"STDERR:\n", stderr.strip())

        except subprocess.TimeoutExpired:
            print(f"⚠ Timeout processing {file_path}")
            if process:
                process.kill()
        except Exception as e:
            if process:
                process.kill()
            print(f"✗ Exception processing {file_path}: {e}")

        self.cleanup_process(file_path)

    def cancel_process(self, file_path):
        """
        Cancel a running process for a specific file.
        """
        with self.lock:
            if file_path in self.running_processes:
                future = self.running_processes[file_path]
                future.cancel()
                del self.running_processes[file_path]

    def cleanup_process(self, file_path):
        """
        Clean up completed process from tracking dictionary.
        """
        #print("cleaning... ")
        with self.lock:
            if file_path in self.running_processes:
                del self.running_processes[file_path]

    def shutdown(self):
        """Shutdown the thread pool executor and cancel pending tasks."""
        print("Shutting down watchdog...")
        self.executor.shutdown(wait=True)


def main():
    """Main function to set up and run the folder watchdog."""
    if len(sys.argv) < 3:
        print("Usage: python folder_watchdog.py <folder_to_watch> <script_to_run>")
        print("Example: python folder_watchdog.py /path/to/watch process_file.py")
        sys.exit(1)

    watch_folder = sys.argv[1]
    script_to_run = sys.argv[2]

    # Validate inputs
    if not os.path.exists(watch_folder):
        print(f"Error: Watch folder does not exist: {watch_folder}")
        sys.exit(1)

    if not os.path.isdir(watch_folder):
        print(f"Error: Watch path is not a directory: {watch_folder}")
        sys.exit(1)

    if not os.path.exists(script_to_run):
        print(f"Error: Script to run does not exist: {script_to_run}")
        sys.exit(1)

    # Convert to absolute paths
    watch_folder = os.path.abspath(watch_folder)
    script_to_run = os.path.abspath(script_to_run)

    print(f"Watching folder: {watch_folder}")
    print(f"Running script: {script_to_run}")
    print("Maximum concurrent processes: 4")
    print("Press Ctrl+C to stop...")

    # Set up file system event handler
    event_handler = FileChangeHandler(script_to_run, max_workers=4)

    # Set up observer
    observer = Observer()
    observer.schedule(event_handler, watch_folder, recursive=True)

    try:
        # Start monitoring
        observer.start()

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nReceived interrupt signal...")

    finally:
        # Clean shutdown
        observer.stop()
        event_handler.shutdown()
        observer.join()
        print("Watchdog stopped.")


if __name__ == "__main__":
    main()

import os
import platform
import signal
import subprocess
import time


class GazeboManager:
    """Manages PX4 SITL and Gazebo processes in a cross-platform way"""

    def __init__(self, px4_dir=None):
        self.px4_dir = px4_dir if px4_dir else os.path.expanduser("~/PX4-Autopilot")
        self.processes = []
        self.running = False
        self.is_windows = platform.system() == "Windows"

        # Verify directory exists before attempting to use it
        if not os.path.exists(self.px4_dir):
            raise FileNotFoundError(f"PX4 directory not found: {self.px4_dir}")

        print(f"Initialized GazeboManager with PX4 directory: {self.px4_dir}")

    def kill_existing_processes(self):
        """Check for and kill any existing PX4 or Gazebo processes"""
        print("Checking for existing PX4/Gazebo processes...")

        try:
            if self.is_windows:
                # Windows commands to find and kill processes
                try:
                    # Check for PX4 processes
                    px4_check = subprocess.run("tasklist | findstr px4", shell=True, capture_output=True, text=True)
                    if px4_check.returncode == 0:
                        print("Found existing PX4 processes, terminating...")
                        subprocess.run("taskkill /F /IM px4.exe", shell=True, stderr=subprocess.PIPE)
                    else:
                        print("No PX4 processes found")

                    # Check for Gazebo processes
                    for proc in ["gz.exe", "gzserver.exe", "gzclient.exe"]:
                        proc_check = subprocess.run(f"tasklist | findstr {proc}", shell=True, capture_output=True,
                                                    text=True)
                        if proc_check.returncode == 0:
                            print(f"Found existing {proc} processes, terminating...")
                            subprocess.run(f"taskkill /F /IM {proc}", shell=True, stderr=subprocess.PIPE)
                        else:
                            print(f"No {proc} processes found")
                except Exception as e:
                    print(f"Warning: Error while checking/killing processes: {e}")
            else:
                # Linux/Mac commands
                try:
                    # Check and kill PX4 processes
                    px4_pids = subprocess.run("pgrep -f px4", shell=True, capture_output=True, text=True)
                    if px4_pids.stdout:
                        print("Found existing PX4 processes:")
                        for pid in px4_pids.stdout.strip().split("\n"):
                            if pid:
                                try:
                                    # Get process info before killing it
                                    proc_info = subprocess.run(f"ps -p {pid} -o cmd=", shell=True,
                                                               capture_output=True, text=True)
                                    print(f"  PID {pid}: {proc_info.stdout.strip()}")
                                    os.kill(int(pid), signal.SIGKILL)
                                    print(f"  Killed PX4 process {pid}")
                                except Exception as e:
                                    print(f"  Failed to kill PX4 process {pid}: {e}")
                    else:
                        print("No PX4 processes found")

                    # Check and kill Gazebo processes
                    for proc in ["gz", "gzserver", "gzclient"]:
                        gz_pids = subprocess.run(f"pgrep -f {proc}", shell=True, capture_output=True, text=True)
                        if gz_pids.stdout:
                            print(f"Found existing {proc} processes, terminating...")
                            subprocess.run(f"pkill -9 {proc}", shell=True, stderr=subprocess.PIPE)
                        else:
                            print(f"No {proc} processes found")
                except Exception as e:
                    print(f"Warning: Error while checking/killing processes: {e}")
        finally:
            # Always wait a moment for processes to fully terminate
            time.sleep(2)
            print("Process cleanup completed")

    def start(self):
        """Start PX4 SITL and Gazebo"""
        if self.running:
            print("Gazebo is already running")
            return

        # Process to be used in finally block
        process = None

        try:
            # Kill any existing processes first
            self.kill_existing_processes()

            print("Starting PX4 SITL and Gazebo...")

            # Test if we can access the directory
            if self.is_windows:
                test_cmd = f"dir {self.px4_dir}"
            else:
                test_cmd = f"cd {self.px4_dir} && ls -la"

            test_result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
            if test_result.returncode != 0:
                print(f"Error accessing PX4 directory: {test_result.stderr}")
                raise Exception(f"Cannot access PX4 directory: {self.px4_dir}")

            if self.is_windows:
                # Windows command might be different depending on PX4 setup
                cmd = f"cd /d {self.px4_dir} && python Tools\\sitl_run.py --model x500"
                kwargs = {}
            else:
                # Linux/Mac command
                cmd = f"cd {self.px4_dir} && make px4_sitl gz_x500"
                kwargs = {"preexec_fn": os.setsid}

            print(f"Running command: {cmd}")
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **kwargs
            )
            self.processes.append(process)

            # Wait for startup and check for errors
            time.sleep(5)

            # Check if process is still running
            if process.poll() is not None:
                # Process exited, get error output
                stdout, stderr = process.communicate()
                stdout_str = stdout.decode('utf-8')
                stderr_str = stderr.decode('utf-8')
                print(f"Gazebo process exited early with code {process.returncode}")
                print(f"STDOUT: {stdout_str}")
                print(f"STDERR: {stderr_str}")

                # Check for specific error about already running
                if "already running" in stdout_str:
                    print("PX4 server is already running. Attempting to stop and restart...")
                    self.stop()
                    # Try one more time after killing processes
                    self.kill_existing_processes()
                    return self.start_without_recursion()

                raise Exception("Gazebo process failed to start")

            # Wait additional time for full initialization
            print("Initial PX4/Gazebo process started, waiting for full initialization...")
            time.sleep(5)

            self.running = True
            print("PX4 SITL and Gazebo started")

        except Exception as e:
            print(f"Failed to start PX4 SITL and Gazebo: {e}")
            # Clean up in the finally block
            raise
        finally:
            # If there was an exception and we're not successfully running,
            # ensure all processes are cleaned up
            if not self.running:
                print("Cleaning up after failed start...")
                if process and process in self.processes:
                    self.processes.remove(process)
                self.stop()

    def start_without_recursion(self):
        """Start without recursive calls - used after killing processes"""
        process = None
        try:
            if self.is_windows:
                cmd = f"cd /d {self.px4_dir} && python Tools\\sitl_run.py --model x500"
                kwargs = {}
            else:
                cmd = f"cd {self.px4_dir} && make px4_sitl gz_x500"
                kwargs = {"preexec_fn": os.setsid}

            print(f"Retrying command: {cmd}")
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **kwargs
            )
            self.processes.append(process)

            # Wait for startup and check for errors
            time.sleep(10)  # Wait a bit longer this time

            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"Second attempt: Gazebo process exited with code {process.returncode}")
                print(f"STDOUT: {stdout.decode('utf-8')}")
                print(f"STDERR: {stderr.decode('utf-8')}")
                raise Exception("Gazebo process failed to start on second attempt")

            self.running = True
            print("PX4 SITL and Gazebo started on second attempt")
            return True

        except Exception as e:
            print(f"Failed on second attempt: {e}")
            raise
        finally:
            # If there was an exception and we're not successfully running,
            # clean up processes
            if not self.running:
                print("Cleaning up after failed second start...")
                if process and process in self.processes:
                    self.processes.remove(process)
                self.stop()

    def stop(self):
        """Stop all processes in a cross-platform way"""
        if not self.running and not self.processes:
            return

        print("Stopping PX4 SITL and Gazebo...")

        try:
            # First try graceful termination
            for process in self.processes:
                try:
                    if process.poll() is None:  # Only if process is still running
                        if self.is_windows:
                            # Windows process termination
                            subprocess.run(f"taskkill /F /T /PID {process.pid}", shell=True)
                        else:
                            # Linux/Mac process termination
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            process.wait(timeout=5)
                except Exception as e:
                    print(f"Error during initial termination: {e}")

            # Then do forceful termination for any that didn't terminate gracefully
            for process in self.processes:
                try:
                    if process.poll() is None:  # If still running after graceful termination
                        if self.is_windows:
                            # Force kill if still running
                            subprocess.run(f"taskkill /F /T /PID {process.pid}", shell=True)
                        else:
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except Exception as e:
                    print(f"Error during forceful termination: {e}")
        finally:
            # Always kill any remaining PX4/Gazebo processes
            self.kill_existing_processes()

            # Always clear the process list and reset running status
            self.processes = []
            self.running = False
            print("PX4 SITL and Gazebo stopped")
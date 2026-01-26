#!/usr/bin/env python3

"""
Wrapper script for full_env_start.sh with user-friendly GUI.
Shows throbber, current step, and last 5 lines of output.
"""

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# Colors for output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
NC = '\033[0m'  # No Color

# Throbber characters
THROBBER_CHARS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

# Step extraction regex
STEP_RE = re.compile(r'STARTING STEP \d+:\s*(.+)')


class InstallGUI:
    def __init__(self):
        self.script_dir = Path(__file__).parent.absolute()
        self.main_script = self.script_dir / 'full_env_start.sh'
        self.log_file = self.script_dir / 'build_codesign.log'
        
        self.current_step = "Initializing..."
        self.last_lines = []
        self.throbber_index = 0
        self.process = None
        self.running = True
        self.display_initialized = False
        self.last_output_time = time.time()
        self.gui_started = False
        
        # Number of lines to reserve for display updates
        # Header (3) + step (1) + blank (1) + "Recent Output:" (1) + separator (1) + 
        # output lines (5) + separator (1) + blank (1) + log path (1) = 15 lines
        self.display_lines = 15
        
        # Thread-safe locks
        self.lock = threading.Lock()
        
    def extract_step(self, line):
        """Extract step name from 'STARTING STEP X: ...' line."""
        match = STEP_RE.search(line)
        if match:
            return match.group(1).strip()
        return None
    
    def init_display(self):
        """Initialize the display - clear screen and print header once."""
        # Clear terminal for clean live progress display
        os.system('clear' if os.name != 'nt' else 'cls')
        
        # Print the header
        sys.stdout.write(f"{CYAN}{'═' * 63}{NC}\n")
        sys.stdout.write(f"{GREEN}CodeSign Installation Progress{NC}\n")
        sys.stdout.write(f"{CYAN}{'═' * 63}{NC}\n")
        sys.stdout.write("\n")
        sys.stdout.write(f"{YELLOW}⠋{NC} Initializing...\n")
        sys.stdout.write("\n")
        sys.stdout.write(f"{CYAN}Recent Output:{NC}\n")
        sys.stdout.write(f"{CYAN}{'─' * 63}{NC}\n")
        # Reserve 5 lines for output
        sys.stdout.write("\n" * 5)
        sys.stdout.write(f"{CYAN}{'─' * 63}{NC}\n")
        sys.stdout.write("\n")
        sys.stdout.write(f"{CYAN}Full log: {self.log_file}{NC}\n")
        sys.stdout.flush()
        
        self.display_initialized = True
    
    def update_display(self):
        """Update the display with current status using cursor movement."""
        with self.lock:
            step = self.current_step
            lines = self.last_lines.copy()
            throbber = THROBBER_CHARS[self.throbber_index]
        
        if not self.display_initialized:
            # Clear screen first to remove any password prompt text
            os.system('clear' if os.name != 'nt' else 'cls')
            self.init_display()
        
        # After init_display, cursor is at bottom (after "Full log:")
        # Move up 12 lines to reach step line (line 5 from top)
        # Structure: header(3) + blank(1) + step(1) + blank(1) + "Recent Output:"(1) + 
        # separator(1) + output(5) + separator(1) + blank(1) + log(1) = 16 total
        sys.stdout.write(f"\033[12F")
        
        # Update step line (line 5)
        sys.stdout.write(f"\033[K{YELLOW}{throbber}{NC} {step}\n")
        
        # We're now at line 6 (blank). Need to get to line 9 (first output line)
        # Line 7 = "Recent Output:", line 8 = separator, line 9 = first output
        # Move down 3 lines by writing newlines (but don't write content, just move)
        sys.stdout.write("\033[K\n")  # Line 6: clear and move down
        sys.stdout.write("\033[K\n")  # Line 7: clear "Recent Output:" line and move down  
        sys.stdout.write("\033[K\n")  # Line 8: clear separator and move down
        
        # Now at line 9 (first output line) - update all 5 output lines
        output_lines = lines[-5:] if lines else ["Waiting for output..."]
        for i in range(5):
            sys.stdout.write("\033[K")  # Clear line
            if i < len(output_lines):
                # Truncate very long lines to avoid wrapping issues
                display_line = output_lines[i].rstrip()[:100]
                sys.stdout.write(display_line)
            sys.stdout.write("\n")
        
        sys.stdout.flush()
    
    def monitor_output(self):
        """Monitor process output and update state."""
        if not self.process:
            return
        
        trigger_line = "Thank you for entering your sudo password if prompted."
        
        # Open log file for appending
        with open(self.log_file, 'a', encoding='utf-8', errors='replace') as log_f:
            # Write header
            log_f.write(f"=== Installation started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log_f.flush()
            
            # Read output line by line
            try:
                for line in iter(self.process.stdout.readline, ''):
                    if not line:
                        break
                    
                    line = line.rstrip()
                    
                    # Write to log
                    log_f.write(line + '\n')
                    log_f.flush()
                    
                    # Before GUI starts, print everything verbatim
                    if not self.gui_started:
                        print(line)
                        sys.stdout.flush()
                        
                        # Check if we've seen the trigger line
                        if trigger_line in line:
                            self.gui_started = True
                            # Clear screen and start GUI
                            os.system('clear' if os.name != 'nt' else 'cls')
                            self.init_display()
                            continue
                    
                    # After GUI starts, update GUI state
                    if self.gui_started:
                        # Update last output time
                        with self.lock:
                            self.last_output_time = time.time()
                        
                        # Update step if this is a "STARTING STEP" line
                        step_name = self.extract_step(line)
                        if step_name:
                            with self.lock:
                                self.current_step = f"Step: {step_name}"
                        
                        # Update last lines buffer
                        with self.lock:
                            self.last_lines.append(line)
                            if len(self.last_lines) > 5:
                                self.last_lines.pop(0)
            except Exception as e:
                # Handle any read errors gracefully
                with self.lock:
                    self.last_lines.append(f"Error reading output: {e}")
                    if len(self.last_lines) > 5:
                        self.last_lines.pop(0)
    
    def run(self, extra_args=None):
        """Run the installation with GUI."""
        if extra_args is None:
            extra_args = []
        
        # Build command - always pass --full
        cmd = [str(self.main_script), '--full'] + extra_args
        
        # Check if script exists
        if not self.main_script.exists():
            print(f"Error: {self.main_script} not found")
            return 1
        
        # Make sure script is executable
        os.chmod(self.main_script, 0o755)
        
        # Initialize log file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Start the process with regular pipes
        # Sudo will write password prompts to /dev/tty directly, which is fine
        # stdin=None so it inherits from terminal (allows sudo to read password)
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=None,  # Inherit from terminal so sudo can read password
                universal_newlines=True,
                bufsize=1,
                cwd=str(self.script_dir),
                encoding='utf-8',
                errors='replace'
            )
        except Exception as e:
            print(f"Error starting installation: {e}")
            return 1
        
        # Start output monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_output, daemon=True)
        monitor_thread.start()
        
        # Main display loop (only runs after GUI starts)
        try:
            while self.process.poll() is None:
                # Wait for GUI to start before updating display
                if not self.gui_started:
                    time.sleep(0.1)
                    continue
                
                # Update throbber
                with self.lock:
                    self.throbber_index = (self.throbber_index + 1) % len(THROBBER_CHARS)
                
                # Update display
                self.update_display()
                
                # Small delay - slightly longer to reduce flicker
                time.sleep(0.15)
            
            # Wait for monitor thread to finish reading remaining output
            monitor_thread.join(timeout=2.0)
            
            # Final display update
            self.update_display()
            
            # Get exit status
            exit_status = self.process.returncode
            
        except KeyboardInterrupt:
            print("\n[CTRL-C] Stopping installation...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            exit_status = 1
        
        # Final display
        os.system('clear' if os.name != 'nt' else 'cls')
        print(f"{CYAN}{'═' * 63}{NC}")
        
        if exit_status == 0:
            print(f"{GREEN}✓ Installation completed successfully!{NC}")
        else:
            print(f"{YELLOW}✗ Installation completed with errors (exit code: {exit_status}){NC}")
        
        print(f"{CYAN}{'═' * 63}{NC}")
        print()
        print(f"{CYAN}Full log: {self.log_file}{NC}")
        print()
        
        # Show last 10 lines of log for context if there was an error
        if exit_status != 0 and self.log_file.exists():
            print(f"{YELLOW}Last 10 lines of output:{NC}")
            print(f"{CYAN}{'─' * 63}{NC}")
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        print(line.rstrip())
            except Exception as e:
                print(f"Error reading log: {e}")
            print(f"{CYAN}{'─' * 63}{NC}")
        
        return exit_status


def main():
    """Main entry point."""
    # Get extra arguments (everything after script name)
    extra_args = sys.argv[1:]
    
    gui = InstallGUI()
    exit_code = gui.run(extra_args)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

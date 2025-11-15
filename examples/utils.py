import subprocess
import sys
import time
import threading
import re
from collections import deque
from typing import Optional, List, Dict

# Global state
server_process: Optional[subprocess.Popen] = None
server_logs = deque(maxlen=1000)
log_lock = threading.Lock()
stdout_thread: Optional[threading.Thread] = None


def read_logs(pipe, log_type='stdout'):
    """Read logs from pipe and save them."""
    try:
        for line in iter(pipe.readline, ''):
            if line:
                with log_lock:
                    server_logs.append((log_type, line.rstrip()))
    except:
        pass
    finally:
        pipe.close()


def start_server(host='0.0.0.0', port=5210, kill_existing=True):
    """
    Start server in background.
    
    Args:
        host: Host (default: '0.0.0.0')
        port: Port (default: 5210)
        kill_existing: Kill existing server on same port (default: True)
    
    Returns:
        Server process object or None
    """
    global server_process, stdout_thread
    
    # Kill existing server if needed
    if kill_existing:
        try:
            subprocess.run(
                ['pkill', '-f', f'kbbridge.server.*{port}'],
                check=False,
                capture_output=True
            )
            time.sleep(1)
        except:
            pass
    
    server_logs.clear()
    
    # Start server
    server_process = subprocess.Popen(
        [sys.executable, '-m', 'kbbridge.server', '--host', host, '--port', str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Start log reader thread
    stdout_thread = threading.Thread(
        target=read_logs,
        args=(server_process.stdout, 'stdout'),
        daemon=True
    )
    stdout_thread.start()
    
    print(f"Server started (PID: {server_process.pid})")
    print("Waiting...")
    time.sleep(3)
    
    # Check if running
    if server_process.poll() is None:
        print("✓ Server is running in background")
        print(f"  Access at: http://{host}:{port}/mcp")
        print(f"\n📋 Use show_logs() to view server logs")
        return server_process
    else:
        print("✗ Server failed to start. Check logs with: show_logs()")
        return None


def show_logs(n_lines=50, follow=False):
    """
    Show server logs.
    
    Args:
        n_lines: How many lines to show (default: 50). Use 0 for all.
        follow: Keep showing new logs (default: False)
    """
    with log_lock:
        logs = list(server_logs)
    
    if not logs:
        print("No logs yet. Server may still be starting...")
        return
    
    recent_logs = logs[-n_lines:] if n_lines > 0 else logs
    
    print(f"\n{'='*80}")
    print(f"Server Logs ({len(recent_logs)} of {len(logs)} lines)")
    print(f"{'='*80}\n")
    
    for log_type, line in recent_logs:
        # Color by log level
        if 'ERROR' in line or 'error' in line.lower():
            print(f"🔴 {line}")
        elif 'WARNING' in line or 'WARN' in line:
            print(f"🟡 {line}")
        elif 'INFO' in line:
            print(f"ℹ️  {line}")
        else:
            print(f"   {line}")
    
    print(f"\n{'='*80}")
    print(f"Total log lines: {len(logs)}")
    
    if follow:
        print("\nFollowing logs (press Ctrl+C to stop)...")
        try:
            last_count = len(logs)
            while True:
                time.sleep(1)
                with log_lock:
                    logs = list(server_logs)
                if len(logs) > last_count:
                    new_logs = logs[last_count:]
                    for log_type, line in new_logs:
                        print(f"   {line}")
                    last_count = len(logs)
        except KeyboardInterrupt:
            print("\nStopped following logs")


def clear_logs():
    """Clear logs."""
    with log_lock:
        server_logs.clear()
    print("Logs cleared")


def stop_server(port=5210):
    """
    Stop server.
    
    Args:
        port: Server port (default: 5210)
    """
    if server_process is not None and server_process.poll() is None:
        server_process.terminate()
        server_process.wait()
        print("Server stopped")
        server_process = None
    else:
        # Kill by port
        subprocess.run(
            ['pkill', '-f', f'kbbridge.server.*{port}'],
            check=False
        )
        print("Stopped server")
        server_process = None


def check_server_status():
    """
    Check if server is running.
    
    Returns:
        True if running, False otherwise
    """
    if server_process is not None:
        if server_process.poll() is None:
            print(f"✓ Server is running (PID: {server_process.pid})")
            return True
        else:
            print("✗ Server is not running")
            return False
    else:
        print("Server process not found")
        return False


# Quick helpers
def logs(n=50):
    """Show last n lines: logs(100) shows last 100."""
    show_logs(n_lines=n)


def tail_logs():
    """Show last 20 lines."""
    show_logs(n_lines=20)


def get_server_process():
    """Get server process."""
    return server_process


def get_logs():
    """Get all logs as list of (type, line) tuples."""
    with log_lock:
        return list(server_logs)


def ensure_server_running(port=5566):
    """
    Start server if not running.
    
    Args:
        port: Server port (default: 5566)
    
    Returns:
        True if server is running, False otherwise
    """
    if not check_server_status():
        start_server(port=port, kill_existing=True)
        return True
    else:
        print("Server already running")
        return True


class LogFormatter:
    """Format server logs for readable blog display"""
    
    # Keywords that indicate important log entries
    IMPORTANT_KEYWORDS = [
        'executing', 'starting', 'completed', 'error', 'failed', 
        'retrieval', 'query', 'intention', 'dataset', 'components',
        'timeout', 'credentials', 'backend'
    ]
    
    # Keywords to filter out (too verbose)
    VERBOSE_KEYWORDS = [
        'request timeout', 'verbose mode', 'status:', 'all required',
        'creating components', 'components created'
    ]
    
    @staticmethod
    def parse_log_line(line: str) -> Optional[Dict]:
        """Parse a log line into structured data"""
        # Format: [timestamp] LEVEL     Message
        pattern = r'\[(.*?)\] (INFO|ERROR|WARNING|DEBUG)\s+(.*)'
        match = re.match(pattern, line)
        if match:
            return {
                'timestamp': match.group(1),
                'level': match.group(2),
                'message': match.group(3).strip()
            }
        return None
    
    @staticmethod
    def is_important(log_entry: Dict) -> bool:
        """Check if log entry is important enough to show"""
        message_lower = log_entry['message'].lower()
        
        # Filter out verbose entries
        if any(keyword in message_lower for keyword in LogFormatter.VERBOSE_KEYWORDS):
            return False
        
        # Include important entries
        if any(keyword in message_lower for keyword in LogFormatter.IMPORTANT_KEYWORDS):
            return True
        
        # Include errors and warnings
        if log_entry['level'] in ['ERROR', 'WARNING']:
            return True
        
        return False
    
    @staticmethod
    def format_logs(log_lines: List[str], max_lines: int = 10, show_all: bool = False) -> str:
        """Format logs for display"""
        parsed_logs = []
        for line in log_lines:
            entry = LogFormatter.parse_log_line(line)
            if entry:
                if show_all or LogFormatter.is_important(entry):
                    parsed_logs.append(entry)
        
        if not parsed_logs:
            return "📝 No relevant logs captured"
        
        # Limit number of logs shown
        if len(parsed_logs) > max_lines and not show_all:
            parsed_logs = parsed_logs[-max_lines:]
            header = f"📋 Showing last {max_lines} relevant log entries:\n\n"
        else:
            header = f"📋 Server Logs ({len(parsed_logs)} entries):\n\n"
        
        output = [header]
        for entry in parsed_logs:
            level_emoji = {
                'INFO': 'ℹ️',
                'ERROR': '❌',
                'WARNING': '⚠️',
                'DEBUG': '🔍'
            }.get(entry['level'], '📝')
            
            # Simplify message (remove redundant prefixes)
            message = entry['message']
            message = re.sub(r'^Server log:\s*', '', message)
            
            output.append(f"{level_emoji} **{entry['level']}**: {message}")
        
        return "\n".join(output)
    
    @staticmethod
    def get_recent_logs(count: int = 20) -> str:
        """Get recent logs from server and format them"""
        logs_tuples = get_logs()
        if not logs_tuples:
            return "📝 No logs available"
        
        # Extract just the log lines (second element of tuple)
        log_lines = [line for _, line in logs_tuples]
        
        # Get recent logs
        recent = log_lines[-count:] if len(log_lines) > count else log_lines
        
        return LogFormatter.format_logs(recent, max_lines=count)


# Create a default formatter instance for convenience
log_formatter = LogFormatter()

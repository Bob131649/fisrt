import select
import sys
import termios
import time
import tty


class SpaceKeyController:
    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self.fd = sys.stdin.fileno() if self.enabled else None
        self.old_settings = None

    def __enter__(self):
        if self.enabled:
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def _read_key(self, timeout=0.0):
        if not self.enabled:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        if not readable:
            return None
        key = sys.stdin.read(1)
        if key == "\x03":
            raise KeyboardInterrupt
        return key

    def wait_for_space(self):
        if not self.enabled:
            print("[keyboard] stdin is not a tty; starting immediately. Ctrl-C stops deploy.")
            return
        print("[keyboard] press Space to start deploy")
        while True:
            if self._read_key(timeout=0.1) == " ":
                print("[keyboard] started; press Space again to stop")
                return

    def stop_requested(self):
        return self._read_key(timeout=0.0) == " "

    def sleep_until(self, end_time):
        while True:
            if self.stop_requested():
                return True
            remaining = end_time - time.time()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))

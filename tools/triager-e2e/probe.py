"""Run l2d on this PTY and check that the caller regains its input discipline."""

import os
import subprocess
import sys


def terminal_mode(stream):
    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        mode = wintypes.DWORD()
        ok = ctypes.windll.kernel32.GetConsoleMode(
            wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno())), ctypes.byref(mode))
        if not ok:
            raise OSError("cannot inspect console mode")
        return mode.value
    import termios

    return termios.tcgetattr(stream.fileno())


if __name__ == "__main__":
    if os.name == "nt":
        # A parent application may run cooked input with echo disabled.
        # Restoring a fixed "cooked" bitmask would turn echo back on.
        import ctypes
        import msvcrt
        from ctypes import wintypes

        handle = wintypes.HANDLE(msvcrt.get_osfhandle(sys.stdin.fileno()))
        original = terminal_mode(sys.stdin)
        if not ctypes.windll.kernel32.SetConsoleMode(handle, (original | 2) & ~4):
            sys.exit("cannot set the parent console mode for the restoration check")
    before = (terminal_mode(sys.stdin), terminal_mode(sys.stdout))
    try:
        code = subprocess.run(sys.argv[1:]).returncode
        if (terminal_mode(sys.stdin), terminal_mode(sys.stdout)) != before:
            sys.exit("l2d left the PTY mode changed")
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetConsoleMode(handle, original)
    print("\r\nTRIAGER_PTY_RESTORED", flush=True)
    sys.exit(code)

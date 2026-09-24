"""Key normalisation to Blender-style event names (``E``, ``CTRL``, ``NUMPAD_1``...)."""

from __future__ import annotations

import sys

MODIFIERS = {"CTRL", "SHIFT", "ALT", "OSKEY"}

_SPECIAL = {
    "ctrl": "CTRL", "ctrl_l": "CTRL", "ctrl_r": "CTRL",
    "shift": "SHIFT", "shift_l": "SHIFT", "shift_r": "SHIFT",
    "alt": "ALT", "alt_l": "ALT", "alt_r": "ALT", "alt_gr": "ALT",
    "cmd": "OSKEY", "cmd_l": "OSKEY", "cmd_r": "OSKEY", "super": "OSKEY",
    "tab": "TAB", "enter": "RET", "esc": "ESC", "space": "SPACE", "backspace": "BACK_SPACE",
    "delete": "DEL", "home": "HOME", "end": "END", "page_up": "PAGE_UP", "page_down": "PAGE_DOWN",
    "up": "UP_ARROW", "down": "DOWN_ARROW", "left": "LEFT_ARROW", "right": "RIGHT_ARROW",
    "insert": "INSERT", "caps_lock": "CAPS_LOCK", "menu": "APP", "print_screen": "PRINT_SCREEN",
}

# Numeric keypad (Blender's view keys). X11 reports keysyms, whose values for ASCII letters
# collide with Windows virtual-key codes (keysym 'e' == 0x65 == VK_NUMPAD5), so the tables are
# kept per platform.
X11_KEYPAD_KEYSYMS = {
    **{0xFFB0 + i: f"NUMPAD_{i}" for i in range(10)},
    0xFFAE: "NUMPAD_PERIOD", 0xFFAB: "NUMPAD_PLUS", 0xFFAD: "NUMPAD_MINUS",
    0xFFAA: "NUMPAD_ASTERIX", 0xFFAF: "NUMPAD_SLASH", 0xFF8D: "NUMPAD_ENTER",
}
WIN32_NUMPAD_VK = {
    **{0x60 + i: f"NUMPAD_{i}" for i in range(10)},
    0x6A: "NUMPAD_ASTERIX", 0x6B: "NUMPAD_PLUS", 0x6D: "NUMPAD_MINUS", 0x6E: "NUMPAD_PERIOD",
    0x6F: "NUMPAD_SLASH",
}

_DIGITS = {"0": "ZERO", "1": "ONE", "2": "TWO", "3": "THREE", "4": "FOUR", "5": "FIVE", "6": "SIX",
           "7": "SEVEN", "8": "EIGHT", "9": "NINE"}


def normalize_key(name: str | None = None, char: str | None = None, vk: int | None = None, *,
                  keypad_keysym: int | None = None, system: str | None = None) -> str:
    """Map a platform key description to a stable name.

    ``name`` is a pynput ``Key`` member name (``ctrl_l``), ``char`` the produced character, ``vk``
    the Windows virtual key code, and ``keypad_keysym`` the X11 keysym of the key at the NumLock
    level (so ``KP_1`` is recognised even when NumLock is off and X reports ``KP_End``).
    """
    system = system or sys.platform
    if keypad_keysym is not None and keypad_keysym in X11_KEYPAD_KEYSYMS:
        return X11_KEYPAD_KEYSYMS[keypad_keysym]
    if system == "win32" and vk is not None and vk in WIN32_NUMPAD_VK:
        return WIN32_NUMPAD_VK[vk]
    if name:
        return _SPECIAL.get(name, name.upper())
    if char:
        if len(char) == 1 and ord(char) < 32:
            # Control characters produced while CTRL is held (e.g. '\x1a' for Ctrl+Z).
            return chr(ord(char) + 64)
        if char in _DIGITS:
            return _DIGITS[char]
        if char.isalpha():
            return char.upper()
        return char
    return "UNKNOWN"


def hotkey_label(key: str, modifiers: list[str] | set[str]) -> str:
    order = [m for m in ("CTRL", "SHIFT", "ALT", "OSKEY") if m in modifiers]
    return "+".join([*order, key])

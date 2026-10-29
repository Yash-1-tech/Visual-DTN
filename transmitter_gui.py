#!/usr/bin/env python3
"""
Visual-DTN Optical Transmitter
==============================
Streams an LT-coded payload as an animated sequence of QR codes.

Pipeline per tick (all in memory, no disk I/O, no threads):
    LTEncoder.next_droplet()  ->  segno QR (level L)  ->  PGM bytes  ->  tk.PhotoImage

Requirements:
    pip install segno          (Tkinter ships with most Python installs)
    visual_dtn.py must sit next to this script.

Usage:
    python transmitter_gui.py                 # type or paste text in the box
    python transmitter_gui.py somefile.bin    # pre-load a file
    python transmitter_gui.py --fps 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from typing import Optional, Sequence

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except ImportError:  # pragma: no cover
    sys.exit("Tkinter is not available. Install it (e.g. `sudo apt install python3-tk`).")

try:
    import segno
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency: segno. Install with `pip install segno`.")

try:
    from visual_dtn import LTEncoder
except ImportError:  # pragma: no cover
    sys.exit("Cannot import LTEncoder: place visual_dtn.py in the same folder.")


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SYMBOL_SIZE = 128      # S: bytes per source chunk -> 148-byte VDF -> QR version 7 at level L
QR_SCALE = 8           # pixels per QR module (6-8 recommended)
QR_BORDER = 4          # quiet zone in modules (QR spec minimum is 4)
FPS_MIN, FPS_MAX, FPS_DEFAULT = 5, 30, 15
FPS_WINDOW = 30        # samples used for the measured FPS
MAX_FILE_BYTES = 1_000_000  # K must stay <= 10,000 -> ~1.28 MB max at S=128; keep headroom

DEFAULT_TEXT = (
    "Visual-DTN demo payload. This text is split into chunks, fountain-coded, and "
    "streamed as QR codes. Any ~1.3x K frames recover it, in any order.\n"
) * 8


# ----------------------------------------------------------------------------
# Pure helpers (no Tk dependency, easy to test)
# ----------------------------------------------------------------------------
_PGM_LUT = bytes([255]) + bytes(255)  # module 0 (light) -> white, anything else (dark) -> black


def make_qr_matrix(droplet: bytes) -> Sequence[Sequence[int]]:
    """Build a QR matrix at error level L from raw bytes.

    boost_error=False is essential: segno otherwise upgrades the EC level whenever the
    chosen version has spare capacity, which would make the code denser than needed.
    mode='byte' stops segno from guessing a text mode for binary data.
    """
    qr = segno.make_qr(droplet, error="l", mode="byte", boost_error=False)
    return qr.matrix


def matrix_to_pgm(matrix: Sequence[Sequence[int]], scale: int = QR_SCALE,
                  border: int = QR_BORDER) -> bytes:
    """Render a 0/1 module matrix to binary PGM (P5) bytes, with quiet zone, nearest-neighbour scaled.

    Tk's built-in PPM reader accepts PGM, so this needs no Pillow and no temp files.
    """
    size = len(matrix) + 2 * border
    width_px = size * scale
    pad = b"\xff" * border
    white_row = b"\xff" * width_px
    rows = [white_row] * (border * scale)
    for mrow in matrix:
        px = pad + bytes(mrow).translate(_PGM_LUT) + pad
        # Expand each module horizontally by `scale`, then repeat the row vertically.
        wide = b"".join([b"\xff" * scale if p else b"\x00" * scale for p in px])
        rows.extend([wide] * scale)
    rows.extend([white_row] * (border * scale))
    header = b"P5\n%d %d\n255\n" % (width_px, width_px)
    return header + b"".join(rows)


def build_encoder(payload: bytes) -> LTEncoder:
    return LTEncoder(payload, symbol_size=SYMBOL_SIZE)


# ----------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------
class TransmitterApp:
    def __init__(self, root: tk.Tk, initial_file: Optional[str], fps: int) -> None:
        self.root = root
        root.title("Visual-DTN Optical Transmitter")
        root.configure(padx=10, pady=10)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- transmitter state (all instance-level) ---
        self.encoder: Optional[LTEncoder] = None
        self.payload_label = ""
        self.running = False
        self.emitted = 0
        self.fps_target = max(FPS_MIN, min(FPS_MAX, fps))
        self._after_id: Optional[str] = None
        self._next_t = 0.0
        self._stamps: "deque[float]" = deque(maxlen=FPS_WINDOW)
        self._photo: Optional[tk.PhotoImage] = None  # keep a reference or Tk garbage-collects it

        self._build_ui()
        self._show_blank()

        if initial_file:
            self._load_file(initial_file)
        else:
            self.text.insert("1.0", DEFAULT_TEXT)
            self._load_text()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        side = QR_SCALE * (45 + 2 * QR_BORDER)  # version-7 footprint
        self.qr_label = tk.Label(self.root, bg="white", width=side, height=side,
                                 bd=2, relief="sunken")
        self.qr_label.pack(side="top")

        hud = ttk.LabelFrame(self.root, text="Diagnostics")
        hud.pack(fill="x", pady=(8, 4))
        self.vars = {k: tk.StringVar(value="-") for k in
                     ("payload", "session", "k", "s", "fps", "emitted")}
        rows = [("Payload", "payload"), ("Session ID", "session"), ("Total Blocks (K)", "k"),
                ("Symbol Size (S)", "s"), ("FPS (target / actual)", "fps"),
                ("Frames emitted", "emitted")]
        for i, (title, key) in enumerate(rows):
            ttk.Label(hud, text=title + ":").grid(row=i, column=0, sticky="w", padx=6)
            ttk.Label(hud, textvariable=self.vars[key]).grid(row=i, column=1, sticky="w", padx=6)

        ctl = ttk.Frame(self.root)
        ctl.pack(fill="x", pady=4)
        self.btn = ttk.Button(ctl, text="Start", command=self.toggle, width=10)
        self.btn.pack(side="left")
        ttk.Label(ctl, text="  FPS").pack(side="left")
        self.fps_scale = tk.Scale(ctl, from_=FPS_MIN, to=FPS_MAX, orient="horizontal",
                                  length=180, command=self._on_fps)
        self.fps_scale.set(self.fps_target)
        self.fps_scale.pack(side="left", padx=6)
        ttk.Button(ctl, text="Open file...", command=self._pick_file).pack(side="right")

        ttk.Label(self.root, text="Payload text (then click 'Load text'):").pack(anchor="w")
        self.text = scrolledtext.ScrolledText(self.root, height=5, width=60, wrap="word")
        self.text.pack(fill="x")
        ttk.Button(self.root, text="Load text", command=self._load_text).pack(anchor="e", pady=4)

        self.status = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.status, foreground="#a00").pack(anchor="w")

    def _show_blank(self) -> None:
        n = QR_SCALE * (45 + 2 * QR_BORDER)
        blank = b"P5\n%d %d\n255\n" % (n, n) + b"\xff" * (n * n)
        self._photo = tk.PhotoImage(data=blank, format="PPM")
        self.qr_label.configure(image=self._photo)

    # ------------------------------------------------------------ payloads
    def _set_payload(self, payload: bytes, label: str) -> None:
        try:
            enc = build_encoder(payload)
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Cannot encode payload", str(exc))
            return
        self.encoder = enc
        self.payload_label = label
        self.emitted = 0
        self._stamps.clear()
        self.status.set("")
        self._refresh_hud()
        if not self.running:
            self._show_blank()

    def _load_text(self) -> None:
        data = self.text.get("1.0", "end-1c").encode("utf-8")
        self._set_payload(data, f"text ({len(data)} bytes)")

    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(title="Select payload file")
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        try:
            size = os.path.getsize(path)
            if size > MAX_FILE_BYTES:
                raise ValueError(f"File is {size} bytes; limit is {MAX_FILE_BYTES}.")
            with open(path, "rb") as fh:
                data = fh.read()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot load file", str(exc))
            return
        self._set_payload(data, f"{os.path.basename(path)} ({len(data)} bytes)")

    # ------------------------------------------------------------- control
    def toggle(self) -> None:
        if self.running:
            self._pause()
        else:
            self._start()

    def _start(self) -> None:
        if self.encoder is None:
            self.status.set("Load a payload first.")
            return
        self.running = True
        self.btn.configure(text="Pause")
        self._stamps.clear()
        self._next_t = time.perf_counter()
        self._schedule(0)

    def _pause(self) -> None:
        self.running = False
        self.btn.configure(text="Start")
        self._cancel()

    def _on_fps(self, value: str) -> None:
        self.fps_target = int(float(value))
        self._refresh_hud()

    def _schedule(self, delay_ms: int) -> None:
        self._after_id = self.root.after(delay_ms, self._tick)

    def _cancel(self) -> None:
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    # ---------------------------------------------------------- frame loop
    def _tick(self) -> None:
        """One scheduled frame. Uses .after() with drift compensation; never blocks the UI."""
        self._after_id = None
        if not self.running or self.encoder is None:
            return
        try:
            self._emit_frame()
        except Exception as exc:  # keep the GUI alive, surface the problem
            self._pause()
            self.status.set(f"Error: {exc}")
            return
        interval = 1.0 / self.fps_target
        now = time.perf_counter()
        self._next_t += interval
        if self._next_t < now - interval:  # fell behind: skip catch-up bursts
            self._next_t = now
        self._schedule(max(1, int((self._next_t - now) * 1000)))

    def _emit_frame(self) -> None:
        assert self.encoder is not None
        droplet = self.encoder.next_droplet()
        pgm = matrix_to_pgm(make_qr_matrix(droplet))
        self._photo = tk.PhotoImage(data=pgm, format="PPM")
        self.qr_label.configure(image=self._photo)
        self.emitted += 1
        self._stamps.append(time.perf_counter())
        self._refresh_hud()

    def _measured_fps(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0

    def _refresh_hud(self) -> None:
        v = self.vars
        v["payload"].set(self.payload_label or "-")
        if self.encoder is not None:
            v["session"].set(f"0x{self.encoder.session_id:08X}")
            v["k"].set(str(self.encoder.k))
            v["s"].set(str(self.encoder.symbol_size))
        v["fps"].set(f"{self.fps_target} / {self._measured_fps():.1f}")
        v["emitted"].set(str(self.emitted))

    # ------------------------------------------------------------ shutdown
    def _on_close(self) -> None:
        self.running = False
        self._cancel()
        self.root.destroy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Visual-DTN animated QR transmitter")
    ap.add_argument("file", nargs="?", help="optional payload file to pre-load")
    ap.add_argument("--fps", type=int, default=FPS_DEFAULT, help=f"{FPS_MIN}-{FPS_MAX} (default {FPS_DEFAULT})")
    args = ap.parse_args()
    root = tk.Tk()
    TransmitterApp(root, args.file, args.fps)
    root.mainloop()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Visual-DTN Protocol Engine
==========================
Luby Transform (LT) rateless erasure coding over Visual Droplet Frames (VDF).

Standard library only. No global mutable state: every encoder/decoder owns its state.

Design notes
------------
* The 20-byte VDF header has no length field, so the encoder prepends a 4-byte
  big-endian length to the payload *inside* the coded stream. The decoder strips it
  after reassembly ("trim to original byte length").
* The Robust Soliton CDF is quantized to 32-bit integers and sampled with an integer
  PRNG output, so degree selection does not depend on float comparison at sample time.
* Neighbor selection uses a sparse (dict-backed) partial Fisher-Yates shuffle: O(d)
  per droplet instead of O(K).
"""

from __future__ import annotations

import bisect
import math
import secrets
import struct
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

# ----------------------------------------------------------------------------
# Protocol constants
# ----------------------------------------------------------------------------
MAGIC_SYNC = 0xD751
PROTOCOL_VERSION = 0x01
FRAME_TYPE_DROPLET = 0x01
HEADER_FMT = ">HBBIIHHI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # must be 20
assert HEADER_SIZE == 20

MAX_K = 10_000
MAX_SYMBOL_SIZE = 4096
LENGTH_PREFIX_SIZE = 4
U32 = 0xFFFFFFFF
TWO32 = 1 << 32


# ----------------------------------------------------------------------------
# 1. PRNG
# ----------------------------------------------------------------------------
class PRNG:
    """SplitMix32-style deterministic 32-bit generator (golden-gamma counter + murmur3 finalizer)."""

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = seed & U32

    def next_u32(self) -> int:
        self._state = (self._state + 0x9E3779B9) & U32
        z = self._state
        z = ((z ^ (z >> 16)) * 0x85EBCA6B) & U32
        z = ((z ^ (z >> 13)) * 0xC2B2AE35) & U32
        return (z ^ (z >> 16)) & U32

    def next_below(self, n: int) -> int:
        """Unbiased integer in [0, n) via bounded rejection sampling (deterministic, capped)."""
        if n <= 0:
            raise ValueError("n must be positive")
        if n == 1:
            return 0
        limit = TWO32 - (TWO32 % n)
        for _ in range(64):  # explicit termination guarantee
            r = self.next_u32()
            if r < limit:
                return r % n
        return self.next_u32() % n  # astronomically unlikely, still deterministic


# ----------------------------------------------------------------------------
# 2. Robust Soliton
# ----------------------------------------------------------------------------
class RobustSoliton:
    """Robust Soliton distribution mu(d), d in [1, K], with an integer CDF for sampling."""

    def __init__(self, k: int, c: float = 0.1, delta: float = 0.5) -> None:
        if not (1 <= k <= MAX_K):
            raise ValueError(f"K must be in [1, {MAX_K}]")
        if c <= 0 or not (0 < delta < 1):
            raise ValueError("require c > 0 and 0 < delta < 1")
        self.k = k

        rho = [0.0] * (k + 1)
        rho[1] = 1.0 / k
        for d in range(2, k + 1):
            rho[d] = 1.0 / (d * (d - 1))

        r = c * math.log(k / delta) * math.sqrt(k)
        pivot = int(math.floor(k / r)) if r > 0 else k
        pivot = max(1, min(k, pivot))  # clamp for tiny K
        tau = [0.0] * (k + 1)
        for d in range(1, pivot):
            tau[d] = r / (d * k)
        tau[pivot] = r * max(0.0, math.log(r / delta)) / k

        weights = [rho[d] + tau[d] for d in range(k + 1)]
        beta = sum(weights[1:])
        self.pmf = [0.0] + [w / beta for w in weights[1:]]

        # Quantized cumulative table, cdf[i] is the exclusive upper bound for degree i+1.
        cdf: List[int] = []
        acc = 0.0
        for d in range(1, k + 1):
            acc += self.pmf[d]
            cdf.append(min(TWO32, int(round(acc * TWO32))))
        cdf[-1] = TWO32  # guarantee every u32 maps to a degree
        self._cdf = cdf

    def sample(self, rng: PRNG) -> int:
        u = rng.next_u32()
        idx = bisect.bisect_right(self._cdf, u)
        return min(idx, self.k - 1) + 1


def derive_neighbors(seed: int, k: int, soliton: RobustSoliton) -> List[int]:
    """Deterministically map (seed, K) -> sorted list of d distinct source indices."""
    rng = PRNG(seed)
    d = min(soliton.sample(rng), k)
    swaps: Dict[int, int] = {}
    chosen: List[int] = []
    for i in range(d):
        j = i + rng.next_below(k - i)
        vi = swaps.get(i, i)
        vj = swaps.get(j, j)
        swaps[j] = vi
        swaps[i] = vj
        chosen.append(vj)
    chosen.sort()
    return chosen


# ----------------------------------------------------------------------------
# 3. Frame serializer
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Frame:
    session_id: int
    seed: int
    k: int
    symbol_size: int
    payload: bytes


class FrameSerializer:
    @staticmethod
    def pack(session_id: int, seed: int, k: int, symbol_size: int, payload: bytes) -> bytes:
        if not (1 <= k <= MAX_K):
            raise ValueError("K out of range")
        if not (1 <= symbol_size <= MAX_SYMBOL_SIZE):
            raise ValueError("symbol size out of range")
        if len(payload) != symbol_size:
            raise ValueError("payload length != symbol size")
        crc = zlib.crc32(payload) & U32
        header = struct.pack(
            HEADER_FMT, MAGIC_SYNC, PROTOCOL_VERSION, FRAME_TYPE_DROPLET,
            session_id & U32, seed & U32, k, symbol_size, crc,
        )
        return header + payload

    @staticmethod
    def unpack(data: object) -> Optional[Frame]:
        """Bounds-checked parse. Returns None on any malformed input; never raises."""
        try:
            if not isinstance(data, (bytes, bytearray, memoryview)):
                return None
            data = bytes(data)
            if len(data) < HEADER_SIZE:
                return None
            magic, ver, ftype, sid, seed, k, s, crc = struct.unpack_from(HEADER_FMT, data, 0)
            if magic != MAGIC_SYNC or ver != PROTOCOL_VERSION or ftype != FRAME_TYPE_DROPLET:
                return None
            if not (1 <= k <= MAX_K) or not (1 <= s <= MAX_SYMBOL_SIZE):
                return None
            if len(data) < HEADER_SIZE + s:
                return None
            payload = data[HEADER_SIZE:HEADER_SIZE + s]  # trailing bytes ignored
            if (zlib.crc32(payload) & U32) != crc:
                return None
            return Frame(sid, seed, k, s, payload)
        except Exception:
            return None


# ----------------------------------------------------------------------------
# 4. Encoder
# ----------------------------------------------------------------------------
class LTEncoder:
    def __init__(self, data: bytes, symbol_size: int = 128, session_id: Optional[int] = None,
                 c: float = 0.1, delta: float = 0.5) -> None:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes")
        if not (1 <= symbol_size <= MAX_SYMBOL_SIZE):
            raise ValueError(f"symbol_size must be in [1, {MAX_SYMBOL_SIZE}]")
        stream = struct.pack(">I", len(data)) + bytes(data) if len(data) <= U32 else None
        if stream is None:
            raise ValueError("payload too large")
        k = max(1, -(-len(stream) // symbol_size))
        if k > MAX_K:
            raise ValueError(f"payload needs K={k} > {MAX_K}; increase symbol_size")
        stream += b"\x00" * (k * symbol_size - len(stream))

        self.k = k
        self.symbol_size = symbol_size
        self.session_id = (secrets.randbits(32) if session_id is None else session_id) & U32
        self._chunks = [int.from_bytes(stream[i * symbol_size:(i + 1) * symbol_size], "big")
                        for i in range(k)]
        self._soliton = RobustSoliton(k, c, delta)
        self._seed_base = PRNG(self.session_id).next_u32()
        self._counter = 0

    def droplet_for_seed(self, seed: int) -> bytes:
        acc = 0
        for idx in derive_neighbors(seed, self.k, self._soliton):
            acc ^= self._chunks[idx]
        payload = acc.to_bytes(self.symbol_size, "big")
        return FrameSerializer.pack(self.session_id, seed, self.k, self.symbol_size, payload)

    def next_droplet(self) -> bytes:
        seed = (self._seed_base + self._counter) & U32
        self._counter += 1
        return self.droplet_for_seed(seed)


# ----------------------------------------------------------------------------
# 5. Decoder (peeling / belief propagation)
# ----------------------------------------------------------------------------
class LTDecoder:
    MAX_PENDING_FACTOR = 8  # buffered-equation cap = factor * K + 64

    def __init__(self) -> None:
        self._locked = False
        self._session = 0
        self.k = 0
        self.symbol_size = 0
        self._soliton: Optional[RobustSoliton] = None
        self._recovered: Dict[int, int] = {}
        self._eq_indices: Dict[int, Set[int]] = {}
        self._eq_data: Dict[int, int] = {}
        self._adj: Dict[int, Set[int]] = {}
        self._next_eq = 0
        self._seen_seeds: Set[int] = set()
        self.processed = 0  # valid, in-session frames consumed

    @property
    def complete(self) -> bool:
        return self._locked and len(self._recovered) == self.k

    @property
    def recovered_count(self) -> int:
        return len(self._recovered)

    def ingest(self, frame_bytes: bytes) -> str:
        """Returns one of: invalid, foreign, ignored, duplicate, redundant, buffered, solved, complete."""
        if self.complete:
            return "ignored"
        frame = FrameSerializer.unpack(frame_bytes)
        if frame is None:
            return "invalid"
        if not self._locked:
            self._session, self.k, self.symbol_size = frame.session_id, frame.k, frame.symbol_size
            self._soliton = RobustSoliton(self.k)
            self._locked = True
        elif (frame.session_id, frame.k, frame.symbol_size) != (self._session, self.k, self.symbol_size):
            return "foreign"
        if frame.seed in self._seen_seeds:
            return "duplicate"
        self._seen_seeds.add(frame.seed)
        self.processed += 1

        assert self._soliton is not None
        remaining = set(derive_neighbors(frame.seed, self.k, self._soliton))
        data = int.from_bytes(frame.payload, "big")
        for idx in list(remaining):
            if idx in self._recovered:
                data ^= self._recovered[idx]
                remaining.discard(idx)

        if not remaining:
            return "redundant"
        if len(remaining) == 1:
            self._peel(next(iter(remaining)), data)
            return "complete" if self.complete else "solved"
        if len(self._eq_indices) >= self.MAX_PENDING_FACTOR * self.k + 64:
            return "ignored"  # memory cap
        eid = self._next_eq
        self._next_eq += 1
        self._eq_indices[eid] = remaining
        self._eq_data[eid] = data
        for idx in remaining:
            self._adj.setdefault(idx, set()).add(eid)
        return "buffered"

    def _peel(self, block: int, data: int) -> None:
        """Iterative ripple. Each block is solved at most once, so loop is bounded by K."""
        queue = [(block, data)]
        budget = self.k + 1
        while queue and budget > 0:
            b, val = queue.pop()
            if b in self._recovered:
                continue
            budget -= 1
            self._recovered[b] = val
            for eid in self._adj.pop(b, ()):
                idxs = self._eq_indices.get(eid)
                if idxs is None:
                    continue
                idxs.discard(b)
                self._eq_data[eid] ^= val
                if len(idxs) == 1:
                    (rem,) = idxs
                    d = self._eq_data.pop(eid)
                    del self._eq_indices[eid]
                    adj = self._adj.get(rem)
                    if adj is not None:
                        adj.discard(eid)
                    queue.append((rem, d))
                elif not idxs:
                    del self._eq_indices[eid]
                    del self._eq_data[eid]

    def result(self) -> Optional[bytes]:
        if not self.complete:
            return None
        s = self.symbol_size
        stream = b"".join(self._recovered[i].to_bytes(s, "big") for i in range(self.k))
        (length,) = struct.unpack_from(">I", stream, 0)
        if length > len(stream) - LENGTH_PREFIX_SIZE:
            return None
        return stream[LENGTH_PREFIX_SIZE:LENGTH_PREFIX_SIZE + length]


# ----------------------------------------------------------------------------
# 6. Test harness
# ----------------------------------------------------------------------------
def _prng_bytes(seed: int, n: int) -> bytes:
    rng = PRNG(seed)
    out = bytearray()
    while len(out) < n:
        out += rng.next_u32().to_bytes(4, "big")
    return bytes(out[:n])


def run_simulation(payload: bytes, symbol_size: int, drop_rate: float, chan_seed: int,
                   session_id: Optional[int] = None, verbose: bool = True) -> dict:
    enc = LTEncoder(payload, symbol_size, session_id=session_id)
    dec = LTDecoder()
    chan = PRNG(chan_seed)
    threshold = int(drop_rate * TWO32)
    sent = dropped = 0
    cap = 20 * enc.k + 1000  # hard stop, no infinite loop
    while not dec.complete and sent < cap:
        pkt = enc.next_droplet()
        sent += 1
        if chan.next_u32() < threshold:
            dropped += 1
            continue
        dec.ingest(pkt)
    out = dec.result()
    assert out == payload, "reconstruction mismatch"
    k, m = enc.k, dec.processed
    stats = {"K": k, "sent": sent, "dropped": dropped, "M": m, "epsilon": (m - k) / k}
    if verbose:
        print(f"  Total K            : {k}")
        print(f"  Total Transmitted  : {sent}")
        print(f"  Total Dropped      : {dropped} ({dropped / sent:.1%})")
        print(f"  Processed M        : {m}")
        print(f"  Overhead epsilon   : {stats['epsilon']:.4f}")
    return stats


def main() -> None:
    print("== Main simulation: 20 KB, S=128, 30% drop ==")
    payload = _prng_bytes(0xC0FFEE, 20 * 1024)
    run_simulation(payload, 128, 0.30, chan_seed=0xBADC0DE, session_id=0x12345678)
    print("  RECONSTRUCTION OK")

    print("\n== Averaged over 20 sessions (K~161) ==")
    eps = [run_simulation(payload, 128, 0.30, chan_seed=1000 + i, session_id=77 + i,
                          verbose=False)["epsilon"] for i in range(20)]
    print(f"  mean eps={sum(eps) / len(eps):.3f}  min={min(eps):.3f}  max={max(eps):.3f}")

    print("\n== Edge cases ==")
    for name, data, s in [("empty", b"", 64), ("1 byte", b"\x42", 64),
                          ("exactly 1 chunk", b"a" * 60, 64), ("odd size", _prng_bytes(5, 1000), 37)]:
        run_simulation(data, s, 0.25, chan_seed=9, session_id=1, verbose=False)
        print(f"  {name}: OK")

    print("\n== Defensive checks ==")
    enc = LTEncoder(payload, 128, session_id=5)
    good = enc.next_droplet()
    flipped = bytearray(good); flipped[-1] ^= 0xFF
    bad_magic = bytearray(good); bad_magic[0] ^= 0xFF
    huge_k = bytearray(good); struct.pack_into(">H", huge_k, 0x0C, 20000)
    huge_s = bytearray(good); struct.pack_into(">H", huge_s, 0x0E, 8000)
    for label, blob in [("CRC mismatch", bytes(flipped)), ("bad magic", bytes(bad_magic)),
                        ("truncated", good[:50]), ("empty", b""), ("K too large", bytes(huge_k)),
                        ("S too large", bytes(huge_s)), ("garbage", _prng_bytes(3, 300)),
                        ("not bytes", None)]:
        assert FrameSerializer.unpack(blob) is None, label
        print(f"  rejected: {label}")

    d = LTDecoder()
    assert d.ingest(good) in ("solved", "buffered", "redundant", "complete")
    other = LTEncoder(b"other session", 128, session_id=6).next_droplet()
    assert d.ingest(other) == "foreign"
    print("  rejected: foreign session")
    assert d.ingest(good) == "duplicate"
    print("  ignored : duplicate seed")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()

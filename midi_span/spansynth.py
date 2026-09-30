# SPDX-FileCopyrightText: 2026 AnySynth contributors
# SPDX-FileCopyrightText: 2024-2026 The YourMT3 Authors
# SPDX-License-Identifier: Apache-2.0

"""The active-note MIDI conditioning representation used by SpanSynth V3 SQ.

The model receives one set of up to 128 active notes every 40 milliseconds.
This module keeps the original fine40 categories and seven numeric features.
Files longer than the model's 20.48-second canvas use consecutive canvases.

Decoding reconstructs notes from the represented boundaries and durations.
The representation groups several MIDI programs into each category and does
not encode drum durations. Decoded programs are category representatives and
decoded drums last 100 milliseconds. Tempo events, controllers, and track
organization are outside this note representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import ClassVar

import numpy as np

from .midi import MidiPerformance, Note, read_midi, write_midi


SAMPLE_RATE = 48_000
HOP_LENGTH = 1_920
FRAMES_PER_WINDOW = 512
SLOTS_PER_FRAME = 128
NUMERIC_WIDTH = 7
WINDOW_SAMPLES = FRAMES_PER_WINDOW * HOP_LENGTH
DRUM_DURATION_SAMPLES = SAMPLE_RATE // 10

PADDING = 0
ONSET = 1
SUSTAIN = 2
OFFSET = 3

NUMERIC_FIELDS = (
    "is_drum",
    "category_sin",
    "category_cos",
    "pitch",
    "velocity",
    "boundary",
    "remaining_duration",
)

# Category order and membership match the original fine40 vocabulary.
# The first program in each group is the canonical decoding representative.
FINE40_GROUPS = {
    "Acoustic Piano": (0, 1, 3, 6, 7),
    "Electric Piano": (2, 4, 5),
    "Chromatic Percussion": tuple(range(8, 16)),
    "Organ": tuple(range(16, 24)),
    "Acoustic Guitar": (24, 25),
    "Clean Electric Guitar": (26, 27, 28),
    "Distorted Electric Guitar": (29, 30, 31),
    "Acoustic Bass": (32, 35),
    "Electric Bass": (33, 34, 36, 37, 38, 39),
    "Violin": (40,),
    "Viola": (41,),
    "Cello": (42,),
    "Contrabass": (43,),
    "Orchestral Harp": (46,),
    "Timpani": (47,),
    "String Ensemble": (48, 49, 44, 45),
    "Synth Strings": (50, 51),
    "Choir and Voice": (52, 53, 54),
    "Orchestra Hit": (55,),
    "Trumpet": (56, 59),
    "Trombone": (57,),
    "Tuba": (58,),
    "French Horn": (60,),
    "Brass Section": (61, 62, 63),
    "Soprano/Alto Sax": (64, 65),
    "Tenor Sax": (66,),
    "Baritone Sax": (67,),
    "Oboe": (68,),
    "English Horn": (69,),
    "Bassoon": (70,),
    "Clarinet": (71,),
    "Pipe": (73, 72, 74, 75, 76, 77, 78, 79),
    "Synth Lead": tuple(range(80, 88)),
    "Synth Pad": tuple(range(88, 96)),
    "Singing Voice": (100,),
    "Singing Voice (chorus)": (101,),
    "Drums": (128,),
    "Sitar": (104,),
    "Banjo": (105,),
    "Fiddle": (110,),
}
CATEGORY_NAMES = tuple(FINE40_GROUPS)
CANONICAL_PROGRAMS = tuple(programs[0] for programs in FINE40_GROUPS.values())
PROGRAM_TO_CATEGORY = {
    program: category
    for category, programs in enumerate(FINE40_GROUPS.values())
    for program in programs
}


@dataclass(frozen=True, slots=True)
class SpanSynthEncoding:
    """Full-file conditioning arrays split into fixed 512-frame windows.

    ``numeric`` has shape ``[windows, 512, 128, 7]``. Categorical arrays and
    ``event_valid`` have shape ``[windows, 512, 128]``. ``valid_mask`` has
    shape ``[windows, 512]`` and excludes padding after the file timeline.

    ``physical_note_id`` links repeated active rows belonging to one note,
    including rows on opposite sides of a window boundary. It is reconstruction
    bookkeeping present in the original data path, not a model input. This
    decoder requires those links. No original notes or raw programs are stored.
    """

    kind_id: np.ndarray
    pitch_id: np.ndarray
    numeric: np.ndarray
    category_id: np.ndarray
    physical_note_id: np.ndarray
    event_valid: np.ndarray
    valid_mask: np.ndarray
    length_samples: int

    sample_rate: ClassVar[int] = SAMPLE_RATE

    def __post_init__(self) -> None:
        if not isinstance(self.numeric, np.ndarray) or self.numeric.ndim != 4:
            raise ValueError("numeric must have shape [windows, 512, 128, 7]")
        windows = self.numeric.shape[0]
        if windows < 1:
            raise ValueError("an encoding must contain at least one window")
        slots = (windows, FRAMES_PER_WINDOW, SLOTS_PER_FRAME)
        expected = {
            "kind_id": (slots, np.dtype("int64")),
            "pitch_id": (slots, np.dtype("int64")),
            "numeric": ((*slots, NUMERIC_WIDTH), np.dtype("float32")),
            "category_id": (slots, np.dtype("int64")),
            "physical_note_id": (slots, np.dtype("int64")),
            "event_valid": (slots, np.dtype("bool")),
            "valid_mask": ((windows, FRAMES_PER_WINDOW), np.dtype("bool")),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} must be a NumPy array")
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
        if (
            isinstance(self.length_samples, bool)
            or not isinstance(self.length_samples, (int, np.integer))
            or self.length_samples < 0
        ):
            raise ValueError("length_samples must be a nonnegative integer")
        if self.length_samples > windows * WINDOW_SAMPLES:
            raise ValueError("length_samples exceeds the encoded canvas")
        frame_count = -((-self.length_samples) // HOP_LENGTH)
        expected_valid = (
            np.arange(windows * FRAMES_PER_WINDOW).reshape(windows, FRAMES_PER_WINDOW)
            < frame_count
        )
        if not np.array_equal(self.valid_mask, expected_valid):
            raise ValueError("valid_mask must cover the full-file timeline as one prefix")


class SpanSynthCapacityError(ValueError):
    """A frame exceeds the original capacity; no notes are discarded."""

    def __init__(self, window_index: int, frame_index: int, row_count: int):
        self.window_index = window_index
        self.frame_index = frame_index
        self.row_count = row_count
        super().__init__(
            f"SpanSynth window {window_index}, frame {frame_index} has "
            f"{row_count} active rows; capacity is {SLOTS_PER_FRAME}"
        )


@dataclass(frozen=True, slots=True)
class _ActiveRow:
    kind: int
    pitch: int
    velocity: int
    category: int
    physical_id: int
    boundary: int
    remaining: int
    is_drum: bool


def _encode_duration(samples: np.ndarray) -> np.ndarray:
    """Original float64 log transform with a float32 output coordinate."""

    clipped = np.minimum(samples.astype(np.float64), WINDOW_SAMPLES)
    denominator = math.log1p(WINDOW_SAMPLES / SAMPLE_RATE)
    return (2.0 * np.log1p(clipped / SAMPLE_RATE) / denominator - 1.0).astype(
        np.float32
    )


def _decode_duration(value: float) -> int:
    normalized = (min(1.0, max(-1.0, value)) + 1.0) * 0.5
    samples = SAMPLE_RATE * math.expm1(
        normalized * math.log1p(WINDOW_SAMPLES / SAMPLE_RATE)
    )
    return min(WINDOW_SAMPLES, max(0, math.floor(samples + 0.5)))


def _rows_for_window(
    notes: tuple[Note, ...], window_index: int
) -> list[list[_ActiveRow]]:
    """Rasterize original half-open intervals without truncating active sets."""

    rows: list[list[_ActiveRow]] = [[] for _ in range(FRAMES_PER_WINDOW)]
    origin = window_index * WINDOW_SAMPLES
    for physical_id, note in enumerate(notes):
        onset, offset = note.onset - origin, note.offset - origin
        category = PROGRAM_TO_CATEGORY[note.program]
        if note.is_drum:
            if 0 <= onset < WINDOW_SAMPLES:
                frame = onset // HOP_LENGTH
                rows[frame].append(
                    _ActiveRow(
                        ONSET, note.pitch, note.velocity, category, physical_id,
                        onset - frame * HOP_LENGTH, 0, True,
                    )
                )
            continue
        first = max(0, onset // HOP_LENGTH)
        stop = min(FRAMES_PER_WINDOW, -((-offset) // HOP_LENGTH))
        for frame in range(first, stop):
            start = frame * HOP_LENGTH
            if start <= onset < start + HOP_LENGTH:
                kind, boundary, remaining = ONSET, onset - start, offset - onset
            elif start < offset <= start + HOP_LENGTH:
                kind, boundary, remaining = OFFSET, offset - start, offset - start
            else:
                kind, boundary, remaining = SUSTAIN, 0, offset - start
            rows[frame].append(
                _ActiveRow(
                    kind, note.pitch, note.velocity, category, physical_id,
                    boundary, remaining, False,
                )
            )
    for frame, active in enumerate(rows):
        if len(active) > SLOTS_PER_FRAME:
            raise SpanSynthCapacityError(window_index, frame, len(active))
        active.sort(
            key=lambda row: (
                row.is_drum, row.category, row.pitch, row.kind, row.physical_id
            )
        )
    return rows


def encode_spansynth_notes(performance: MidiPerformance) -> SpanSynthEncoding:
    """Encode sample-grid notes using the V3 SQ fine40 conditioning layout.

    The input clock must be 48 kHz. Every overlapping melodic note occupies
    one row per frame, while each drum occupies only its onset frame. More
    than 128 active rows in any frame raises ``SpanSynthCapacityError``.

    Consecutive windows retain the model's fixed frame and slot dimensions.
    A note crossing a window is not restarted. Its physical ID stays constant.
    """

    if performance.sample_rate != SAMPLE_RATE:
        raise ValueError(f"SpanSynth notes must use a {SAMPLE_RATE} Hz sample clock")
    notes = tuple(
        sorted(
            performance.notes,
            key=lambda note: (
                note.onset, note.offset, note.program, note.pitch, note.velocity
            ),
        )
    )
    for note in notes:
        if note.program not in PROGRAM_TO_CATEGORY:
            raise ValueError(f"program {note.program} is outside the fine40 vocabulary")
        if not note.is_drum and note.offset <= note.onset:
            raise ValueError("melodic notes must have positive duration")
    frame_count = -((-performance.length_samples) // HOP_LENGTH)
    windows = max(1, -((-frame_count) // FRAMES_PER_WINDOW))
    slots = (windows, FRAMES_PER_WINDOW, SLOTS_PER_FRAME)
    kind_id = np.zeros(slots, dtype=np.int64)
    pitch_id = np.zeros(slots, dtype=np.int64)
    numeric = np.zeros((*slots, NUMERIC_WIDTH), dtype=np.float32)
    category_id = np.full(slots, -1, dtype=np.int64)
    physical_note_id = np.full(slots, -1, dtype=np.int64)
    event_valid = np.zeros(slots, dtype=np.bool_)
    valid_mask = (
        np.arange(windows * FRAMES_PER_WINDOW).reshape(windows, FRAMES_PER_WINDOW)
        < frame_count
    )
    for window in range(windows):
        for frame, rows in enumerate(_rows_for_window(notes, window)):
            if not rows or not valid_mask[window, frame]:
                continue
            count = len(rows)
            kind_id[window, frame, :count] = [row.kind for row in rows]
            pitch_id[window, frame, :count] = [row.pitch + 1 for row in rows]
            category_id[window, frame, :count] = [row.category for row in rows]
            physical_note_id[window, frame, :count] = [row.physical_id for row in rows]
            event_valid[window, frame, :count] = True
            for slot, row in enumerate(rows):
                angle = 2.0 * math.pi * row.category / len(FINE40_GROUPS)
                numeric[window, frame, slot, :6] = (
                    1.0 if row.is_drum else -1.0,
                    math.sin(angle),
                    math.cos(angle),
                    2.0 * row.pitch / 127.0 - 1.0,
                    2.0 * row.velocity / 127.0 - 1.0,
                    0.0 if row.kind == SUSTAIN else 2.0 * row.boundary / HOP_LENGTH - 1.0,
                )
            numeric[window, frame, :count, 6] = _encode_duration(
                np.array([row.remaining for row in rows], dtype=np.int64)
            )
    return SpanSynthEncoding(
        kind_id, pitch_id, numeric, category_id, physical_note_id,
        event_valid, valid_mask, int(performance.length_samples),
    )


def encode_spansynth_midi(path: str | Path) -> SpanSynthEncoding:
    """Read a MIDI file and encode its notes on the original 48 kHz clock."""

    return encode_spansynth_notes(read_midi(path, sample_rate=SAMPLE_RATE))


@dataclass(slots=True)
class _DecodedNote:
    category: int
    pitch: int
    velocity: int
    onset: int | None = None
    offset: int = 0


def decode_spansynth_notes(encoding: SpanSynthEncoding) -> MidiPerformance:
    """Recover notes from active rows and their physical-note links.

    Category representatives replace original programs and drums last 100 ms.
    Endpoints use the latest represented row, so a note longer than 20.48 s
    is recovered through its final offset rather than its saturated initial
    duration. Every melodic note in a full-file encoding must have an onset.
    """

    if not isinstance(encoding, SpanSynthEncoding):
        raise TypeError("encoding must be SpanSynthEncoding")
    # Arrays remain editable for callers, so recheck their current layout.
    encoding.__post_init__()
    if not np.isfinite(encoding.numeric).all():
        raise ValueError("numeric coordinates must be finite")
    if np.any(encoding.event_valid & ~encoding.valid_mask[..., None]):
        raise ValueError("invalid frames cannot contain active MIDI rows")
    decoded: dict[int, _DecodedNote] = {}
    for window, frame, slot in np.argwhere(encoding.event_valid):
        position = (window, frame, slot)
        kind = int(encoding.kind_id[position])
        category = int(encoding.category_id[position])
        physical_id = int(encoding.physical_note_id[position])
        pitch = int(encoding.pitch_id[position]) - 1
        values = encoding.numeric[position]
        velocity = math.floor((float(values[4]) + 1.0) * 127.0 / 2.0 + 0.5)
        if (
            kind not in (ONSET, SUSTAIN, OFFSET)
            or not 0 <= category < len(FINE40_GROUPS)
            or physical_id < 0
            or not 0 <= pitch <= 127
            or not 1 <= velocity <= 127
        ):
            raise ValueError("active rows need valid kind, category, ID, pitch and velocity")
        note = decoded.setdefault(physical_id, _DecodedNote(category, pitch, velocity))
        if (note.category, note.pitch, note.velocity) != (category, pitch, velocity):
            raise ValueError(f"physical note {physical_id} changes identity between frames")
        start = (int(window) * FRAMES_PER_WINDOW + int(frame)) * HOP_LENGTH
        boundary = math.floor((float(values[5]) + 1.0) * HOP_LENGTH / 2.0 + 0.5)
        if kind == ONSET and not 0 <= boundary < HOP_LENGTH:
            raise ValueError("onset boundaries must lie inside their frame")
        if kind == OFFSET and not 0 < boundary <= HOP_LENGTH:
            raise ValueError("offset boundaries must lie within (frame start, frame end]")
        if kind == ONSET:
            if note.onset is not None:
                raise ValueError(f"physical note {physical_id} has more than one onset")
            note.onset = start + boundary
        if CANONICAL_PROGRAMS[category] == 128:
            if kind != ONSET:
                raise ValueError("drums must use a single onset row")
            note.offset = start + boundary + DRUM_DURATION_SAMPLES
        elif kind == OFFSET:
            note.offset = start + boundary
        else:
            reference = start + boundary if kind == ONSET else start
            note.offset = reference + _decode_duration(float(values[6]))
    notes: list[Note] = []
    for physical_id, note in decoded.items():
        if note.onset is None:
            raise ValueError(
                f"physical note {physical_id} has no onset in this full-file encoding"
            )
        if note.offset <= note.onset:
            raise ValueError(f"physical note {physical_id} has a nonpositive decoded duration")
        notes.append(
            Note(
                note.onset, note.offset, note.pitch, note.velocity,
                CANONICAL_PROGRAMS[note.category],
            )
        )
    notes.sort(
        key=lambda note: (
            note.onset, note.offset, note.program, note.pitch, note.velocity
        )
    )
    length = max(encoding.length_samples, max((note.offset for note in notes), default=0))
    return MidiPerformance(tuple(notes), SAMPLE_RATE, length)


def decode_spansynth_midi(encoding: SpanSynthEncoding, path: str | Path) -> Path:
    """Decode represented notes and write a canonical MIDI file."""

    return write_midi(decode_spansynth_notes(encoding), path)

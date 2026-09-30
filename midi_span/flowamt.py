# SPDX-FileCopyrightText: 2026 AnySynth contributors
# SPDX-License-Identifier: Apache-2.0

"""Physical ON/OFF endpoints from the small FlowAMT ``v2_offset`` codec.

The default is piano-only. ``program_mode="pgb"`` restores the historical
Piano/Guitar/Bass vocabulary without changing the 32-by-9 frame layout.
There is no duration coordinate, learned decoder, or training objective here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from .midi import MidiPerformance, Note, read_midi, write_midi


SAMPLE_RATE = 16_000
HOP_LENGTH = 256
LANES_PER_BANK = 16
ENDPOINT_COUNT = 32
FEATURES_PER_ENDPOINT = 9
FRAME_WIDTH = ENDPOINT_COUNT * FEATURES_PER_ENDPOINT

# Endpoint type is determined by the bank, rather than generated as a feature.
_ON = 0
_OFF = 1
_PROGRAM_RANGES = ((0, 7), (24, 31), (32, 39))
_PROGRAM_REPRESENTATIVES = (0, 24, 32)

_PROGRAM_CODES = np.asarray(
    ((1.0, 0.0), (-0.5, math.sqrt(3.0) / 2.0),
     (-0.5, -math.sqrt(3.0) / 2.0)),
    dtype=np.float64,
)
_PITCHES = np.arange(128, dtype=np.float64)
_PITCH_ANGLES = _PITCHES * (2.0 * math.pi / 12.0)
_PITCH_CODES = np.column_stack(
    (np.cos(_PITCH_ANGLES), np.sin(_PITCH_ANGLES),
     (_PITCHES - 63.5) / 63.5)
)
_VELOCITIES = np.arange(1, 128, dtype=np.float64)
_VELOCITY_CODES = np.column_stack(
    (2.0 * _VELOCITIES / 127.0 - 1.0,
     2.0 * np.log1p(_VELOCITIES) / math.log(128.0) - 1.0)
)
_SUBSAMPLE_CODES = (
    2.0 * np.arange(HOP_LENGTH, dtype=np.float64) / (HOP_LENGTH - 1) - 1.0
)[:, None]

# The original encoder stores FP32 coordinates but projects against FP64 codes.
_PROGRAM_ENCODING = _PROGRAM_CODES.astype(np.float32)
_PITCH_ENCODING = _PITCH_CODES.astype(np.float32)
_VELOCITY_ENCODING = _VELOCITY_CODES.astype(np.float32)
_SUBSAMPLE_ENCODING = (
    2.0 * np.arange(HOP_LENGTH, dtype=np.float32) / (HOP_LENGTH - 1) - 1.0
)


def _validate_program_mode(program_mode: str) -> None:
    if program_mode not in ("piano", "pgb"):
        raise ValueError("program_mode must be 'piano' or 'pgb'")


@dataclass(frozen=True, slots=True)
class MidiSpanEncoding:
    """FP32 ``[T,288]`` endpoints and the MIDI timeline's sample length.

    Each frame contains 16 ON lanes followed by 16 OFF lanes. An occupied
    endpoint stores presence, two program coordinates, three pitch coordinates,
    two velocity coordinates, and its sample position inside the frame.
    Empty endpoints are ``[-1,0,0,0,0,0,0,0,0]``. Each bank uses an occupied
    prefix, ordered by sample position, program group, pitch, and velocity.

    The canvas includes the sample at ``length_samples``, so a final physical
    OFF endpoint is retained even when it falls on a frame boundary.
    """

    values: np.ndarray
    length_samples: int
    program_mode: str = "piano"
    sample_rate: ClassVar[int] = SAMPLE_RATE
    hop_length: ClassVar[int] = HOP_LENGTH

    def __post_init__(self) -> None:
        _validate_program_mode(self.program_mode)
        if not isinstance(self.values, np.ndarray):
            raise TypeError("values must be a NumPy array")
        if self.values.dtype != np.float32:
            raise TypeError("values must have dtype float32")
        if self.values.ndim != 2 or self.values.shape[1] != FRAME_WIDTH:
            raise ValueError("values must have shape [T, 288]")
        if not np.isfinite(self.values).all():
            raise ValueError("values must contain only finite coordinates")
        if isinstance(self.length_samples, bool) or not isinstance(
            self.length_samples, int
        ):
            raise TypeError("length_samples must be an integer")
        if self.length_samples < 0:
            raise ValueError("length_samples must be nonnegative")
        if self.length_samples >= self.values.shape[0] * HOP_LENGTH:
            raise ValueError("the frame canvas must include length_samples")

    @property
    def features(self) -> np.ndarray:
        """View ``values`` as ``[T,32,9]`` without copying."""

        return self.values.reshape(-1, ENDPOINT_COUNT, FEATURES_PER_ENDPOINT)


@dataclass(frozen=True, slots=True)
class _Endpoint:
    sample: int
    bank: int
    program_index: int
    pitch: int
    velocity: int


def _program_index(note: Note, program_mode: str) -> int:
    if note.is_drum:
        raise ValueError("MIDI endpoints do not represent drums")
    ranges = _PROGRAM_RANGES[:1] if program_mode == "piano" else _PROGRAM_RANGES
    for index, (lower, upper) in enumerate(ranges):
        if lower <= note.program <= upper:
            return index
    allowed = "Piano programs 0–7" if program_mode == "piano" else (
        "Piano 0–7, Guitar 24–31, or Bass 32–39 programs"
    )
    raise ValueError(f"MIDI endpoints require {allowed}; got program {note.program}")


def canonicalize_notes(
    performance: MidiPerformance, *, program_mode: str = "piano"
) -> MidiPerformance:
    """Map programs and turn same-pitch overlaps into explicit reattack notes.

    This applies the PoC's activity-union/reattack preprocessing rule. After
    mapping programs, notes with an identical program, pitch, and onset merge
    using their maximum offset and velocity. Each connected activity component
    is then split at every distinct onset. The velocity of each resulting
    interval is the maximum among merged notes active at that interval's start.
    Touching notes retain their reattack and gaps remain silent.

    This changes overlapping note intervals before encoding. The returned
    performance is the intended roundtrip target, with the original timeline
    length. Only exact-onset merging is applied here; the historical
    source-specific near-onset doubling rule requires source identities.
    """

    _validate_program_mode(program_mode)
    if not isinstance(performance, MidiPerformance):
        raise TypeError("performance must be a MidiPerformance")
    if performance.sample_rate != SAMPLE_RATE:
        raise ValueError("MIDI endpoints require notes quantized at 16000 Hz")

    tracks: dict[tuple[int, int], dict[int, Note]] = {}
    for note in performance.notes:
        program = _PROGRAM_REPRESENTATIVES[_program_index(note, program_mode)]
        onsets = tracks.setdefault((program, note.pitch), {})
        previous = onsets.get(note.onset)
        onsets[note.onset] = Note(
            onset=note.onset,
            offset=max(note.offset, previous.offset) if previous else note.offset,
            pitch=note.pitch,
            velocity=max(note.velocity, previous.velocity) if previous else note.velocity,
            program=program,
        )

    notes: list[Note] = []

    def append_component(component: list[Note]) -> None:
        boundaries = sorted({note.onset for note in component})
        boundaries.append(max(note.offset for note in component))
        for onset, offset in zip(boundaries, boundaries[1:]):
            velocity = max(
                note.velocity for note in component
                if note.onset <= onset < note.offset
            )
            notes.append(Note(
                onset, offset, component[0].pitch, velocity, component[0].program
            ))

    for onsets in tracks.values():
        ordered = sorted(onsets.values(), key=lambda note: note.onset)
        component = [ordered[0]]
        component_end = ordered[0].offset
        for note in ordered[1:]:
            if note.onset <= component_end:
                component.append(note)
                component_end = max(component_end, note.offset)
            else:
                append_component(component)
                component = [note]
                component_end = note.offset
        append_component(component)

    notes.sort(key=lambda note: (
        note.onset, note.offset, note.program, note.pitch, note.velocity,
    ))
    return MidiPerformance(
        tuple(notes), performance.sample_rate, performance.length_samples
    )


def encode_notes(
    performance: MidiPerformance, *, program_mode: str = "piano",
    overlap_policy: str = "reject",
) -> MidiSpanEncoding:
    """Encode sample-quantized notes as ordered physical ON/OFF endpoints.

    Piano programs 0–7 decode to program 0. With ``program_mode="pgb"``,
    programs 24–31 decode to 24 and programs 32–39 decode to 32. Other
    programs and drums are rejected. With the default ``overlap_policy="reject"``,
    notes sharing a mapped program and pitch must not overlap. Adjacent reattacks
    are allowed. ``overlap_policy="reattack"`` first applies
    :func:`canonicalize_notes`, which changes overlapping note intervals
    into a nonoverlapping activity union with explicit reattacks. More than 16
    ON or 16 OFF endpoints in a frame are rejected without truncation.
    """

    _validate_program_mode(program_mode)
    if overlap_policy not in ("reject", "reattack"):
        raise ValueError("overlap_policy must be 'reject' or 'reattack'")
    if not isinstance(performance, MidiPerformance):
        raise TypeError("performance must be a MidiPerformance")
    if performance.sample_rate != SAMPLE_RATE:
        raise ValueError("MIDI endpoints require notes quantized at 16000 Hz")
    if overlap_policy == "reattack":
        performance = canonicalize_notes(
            performance, program_mode=program_mode
        )

    mapped = [(_program_index(note, program_mode), note)
              for note in performance.notes]
    previous_offsets: dict[tuple[int, int], int] = {}
    for program_index, note in sorted(
        mapped, key=lambda item: (item[1].onset, item[1].offset)
    ):
        key = (program_index, note.pitch)
        if note.onset < previous_offsets.get(key, 0):
            program = _PROGRAM_REPRESENTATIVES[program_index]
            raise ValueError(
                f"overlapping notes for mapped program {program}, "
                f"pitch {note.pitch} at sample {note.onset}"
            )
        previous_offsets[key] = note.offset

    frame_count = performance.length_samples // HOP_LENGTH + 1
    features = np.zeros(
        (frame_count, ENDPOINT_COUNT, FEATURES_PER_ENDPOINT), dtype=np.float32
    )
    features[..., 0] = -1.0
    groups: dict[tuple[int, int], list[_Endpoint]] = {}
    for program_index, note in mapped:
        for bank, sample in ((_ON, note.onset), (_OFF, note.offset)):
            endpoint = _Endpoint(
                sample, bank, program_index, note.pitch, note.velocity
            )
            groups.setdefault((sample // HOP_LENGTH, bank), []).append(endpoint)

    for (frame, bank), endpoints in groups.items():
        if len(endpoints) > LANES_PER_BANK:
            name = "ON" if bank == _ON else "OFF"
            raise ValueError(
                f"frame {frame} has {len(endpoints)} {name} endpoints; "
                "the maximum is 16"
            )
        endpoints.sort(key=lambda item: (
            item.sample % HOP_LENGTH, item.program_index, item.pitch,
            item.velocity,
        ))
        for rank, endpoint in enumerate(endpoints):
            cell = features[frame, bank * LANES_PER_BANK + rank]
            cell[0] = 1.0
            cell[1:3] = _PROGRAM_ENCODING[endpoint.program_index]
            cell[3:6] = _PITCH_ENCODING[endpoint.pitch]
            cell[6:8] = _VELOCITY_ENCODING[endpoint.velocity - 1]
            cell[8] = _SUBSAMPLE_ENCODING[endpoint.sample % HOP_LENGTH]

    return MidiSpanEncoding(
        features.reshape(frame_count, FRAME_WIDTH),
        performance.length_samples,
        program_mode,
    )


def encode_midi(
    path: str | Path, *, program_mode: str = "piano",
    overlap_policy: str = "reject",
) -> MidiSpanEncoding:
    """Read MIDI at 16000 Hz and encode, with an explicit overlap policy.

    ``"reattack"`` applies :func:`canonicalize_notes` to sounding notes,
    including overlaps caused by the sustain pedal. ``"reject"`` is the default.
    """

    return encode_notes(
        read_midi(path, sample_rate=SAMPLE_RATE), program_mode=program_mode,
        overlap_policy=overlap_policy,
    )


def _nearest_index(value: np.ndarray, codebook: np.ndarray) -> int:
    """Project by squared Euclidean distance, resolving ties downward."""

    distances = np.square(value.astype(np.float64) - codebook).sum(axis=1)
    return int(distances.argmin())


def _decode_endpoints(encoding: MidiSpanEncoding) -> list[_Endpoint]:
    bounded = np.clip(encoding.features, -1.0, 1.0)
    presence = bounded[..., 0].reshape(-1, 2, LANES_PER_BANK)
    prefix_scores = np.concatenate(
        (np.zeros((*presence.shape[:-1], 1), dtype=np.float64),
         np.cumsum(presence, axis=-1, dtype=np.float64)),
        axis=-1,
    )
    counts = prefix_scores.argmax(axis=-1)
    endpoints: list[_Endpoint] = []
    for frame, frame_counts in enumerate(counts):
        for bank, count in enumerate(frame_counts):
            for rank in range(int(count)):
                cell = bounded[frame, bank * LANES_PER_BANK + rank]
                program_index = (
                    0 if encoding.program_mode == "piano" else
                    _nearest_index(cell[1:3], _PROGRAM_CODES)
                )
                endpoints.append(_Endpoint(
                    sample=(frame * HOP_LENGTH
                            + _nearest_index(cell[8:9], _SUBSAMPLE_CODES)),
                    bank=bank,
                    program_index=program_index,
                    pitch=_nearest_index(cell[3:6], _PITCH_CODES),
                    velocity=_nearest_index(cell[6:8], _VELOCITY_CODES) + 1,
                ))
    return endpoints


def decode_notes(encoding: MidiSpanEncoding) -> MidiPerformance:
    """Project coordinates and pair endpoints by mapped program and pitch.

    Prefix count is the argmax of FP64 cumulative presence, including count
    zero. Payloads use nearest legal coordinate codes after clamping to [-1,1].
    The piano-only decoder has one legal program, so generated program
    coordinates do not affect its output.

    Pairing is chronological, with OFF before ON at the same sample. ON
    velocity is authoritative. Lane indices do not connect ON to OFF. Unlike
    the historical permissive materializer, this function raises on orphan
    OFF endpoints, overlapping ON endpoints, or unclosed notes rather than
    silently discarding them. Endpoints beyond the declared timeline also
    raise. Clean supported encodings roundtrip to their mapped note content.
    """

    if not isinstance(encoding, MidiSpanEncoding):
        raise TypeError("encoding must be a MidiSpanEncoding")
    # Values remain editable for experiments; validate again before decoding.
    encoding.__post_init__()
    endpoints = _decode_endpoints(encoding)
    endpoints.sort(key=lambda item: (
        item.sample, item.program_index, item.pitch,
        0 if item.bank == _OFF else 1, item.velocity,
    ))
    open_notes: dict[tuple[int, int], _Endpoint] = {}
    notes: list[Note] = []
    for endpoint in endpoints:
        if endpoint.sample > encoding.length_samples:
            raise ValueError(
                f"endpoint at sample {endpoint.sample} exceeds length_samples"
            )
        key = (endpoint.program_index, endpoint.pitch)
        program = _PROGRAM_REPRESENTATIVES[endpoint.program_index]
        if endpoint.bank == _ON:
            if key in open_notes:
                raise ValueError(
                    f"unclosed ON before another ON for program {program}, "
                    f"pitch {endpoint.pitch} at sample {endpoint.sample}"
                )
            open_notes[key] = endpoint
        else:
            onset = open_notes.pop(key, None)
            if onset is None:
                raise ValueError(
                    f"orphan OFF for program {program}, pitch {endpoint.pitch} "
                    f"at sample {endpoint.sample}"
                )
            if endpoint.sample <= onset.sample:
                raise ValueError("an OFF endpoint must follow its ON endpoint")
            notes.append(Note(
                onset=onset.sample,
                offset=endpoint.sample,
                pitch=onset.pitch,
                velocity=onset.velocity,
                program=program,
            ))
    if open_notes:
        onset = next(iter(open_notes.values()))
        program = _PROGRAM_REPRESENTATIVES[onset.program_index]
        raise ValueError(
            f"unclosed ON for program {program}, pitch {onset.pitch} "
            f"at sample {onset.sample}"
        )
    notes.sort(key=lambda note: (
        note.onset, note.offset, note.program, note.pitch, note.velocity,
    ))
    return MidiPerformance(
        tuple(notes), sample_rate=SAMPLE_RATE,
        length_samples=encoding.length_samples,
    )


def decode_midi(encoding: MidiSpanEncoding, path: str | Path) -> Path:
    """Decode an endpoint encoding and write a canonical MIDI file."""

    return write_midi(decode_notes(encoding), path)


__all__ = [
    "MidiSpanEncoding",
    "canonicalize_notes",
    "encode_notes",
    "encode_midi",
    "decode_notes",
    "decode_midi",
]

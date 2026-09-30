# SPDX-FileCopyrightText: 2026 AnySynth contributors
# SPDX-FileCopyrightText: 2024-2026 The YourMT3 Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared, tempo-aware MIDI I/O on an integer sample clock.

Notes describe sounding intervals: sustain pedal extends melodic offsets.
MIDI tracks, tempo events and controllers are consumed, not stored in spans.
The MIDI writer emits a canonical arrangement with sample-accurate timing.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import mido


@dataclass(frozen=True, slots=True)
class Note:
    """One half-open interval [onset, offset), measured in integer samples.

    Programs 0..127 are zero-based GM programs; 128 means percussion.
    Velocity is the note-on velocity, not the release velocity.
    """

    onset: int
    offset: int
    pitch: int
    velocity: int
    program: int = 0

    def __post_init__(self) -> None:
        for name in ("onset", "offset", "pitch", "velocity", "program"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.onset < 0 or self.offset <= self.onset:
            raise ValueError("notes require 0 <= onset < offset")
        if not 0 <= self.pitch <= 127:
            raise ValueError("pitch must be in 0..127")
        if not 1 <= self.velocity <= 127:
            raise ValueError("velocity must be in 1..127")
        if not 0 <= self.program <= 128:
            raise ValueError("program must be in 0..128 (128 is percussion)")

    @property
    def is_drum(self) -> bool:
        return self.program == 128


@dataclass(frozen=True, slots=True)
class MidiPerformance:
    """Notes and file length on a shared sample clock, including trailing silence."""

    notes: tuple[Note, ...]
    sample_rate: int
    length_samples: int

    def __post_init__(self) -> None:
        if not isinstance(self.notes, tuple) or not all(
            isinstance(note, Note) for note in self.notes
        ):
            raise TypeError("notes must be a tuple of Note objects")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if isinstance(self.length_samples, bool) or not isinstance(self.length_samples, int):
            raise TypeError("length_samples must be an integer")
        if self.length_samples < max((note.offset for note in self.notes), default=0):
            raise ValueError("length_samples must contain every note")


def _note_key(note: Note) -> tuple[int, ...]:
    return note.onset, note.offset, note.program, note.pitch, note.velocity


def read_midi(path: str | Path, *, sample_rate: int) -> MidiPerformance:
    """Read type-0/1 MIDI, resolving tempo, program changes and sustain.

    An unset program defaults to GM piano (0); channel 9 is percussion (128).
    Repeated key-down notes pair with key-up events in FIFO order per port,
    channel and pitch. Unmatched key-ups are ignored; hanging notes close at
    file end. Sample times use floor(seconds * sample_rate + 0.5), and an
    interval shorter than one sample is extended to one sample.
    """

    MidiPerformance((), sample_rate, 0)  # Validate the clock before reading.
    midi = mido.MidiFile(path)
    if midi.type == 2:
        raise ValueError("type-2 MIDI has no single shared timeline")
    if midi.ticks_per_beat <= 0:
        raise ValueError("SMPTE MIDI timing is unsupported")

    # Retain each track's port while merging ticks; port is track-local MIDI
    # metadata, so a plain merged-message iterator loses that association.
    events = []
    for track_index, track in enumerate(midi.tracks):
        tick, port = 0, 0
        for event_index, message in enumerate(track):
            tick += message.time
            if message.type == "midi_port":
                port = message.port
            events.append((tick, track_index, event_index, port, message))
    events.sort(key=lambda event: event[:3])

    programs: dict[tuple[int, int], int] = defaultdict(int)
    pedal: dict[tuple[int, int], bool] = defaultdict(bool)
    active: dict[tuple[int, int, int], deque[tuple[int, int, int]]] = defaultdict(deque)
    sustained: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    notes: list[Note] = []
    seconds, last_tick, tempo = 0.0, 0, 500_000

    def finish(onset: int, pitch: int, velocity: int, program: int, offset: int) -> None:
        notes.append(Note(onset, max(onset + 1, offset), pitch, velocity, program))

    for tick, _, _, port, message in events:
        seconds += mido.tick2second(tick - last_tick, midi.ticks_per_beat, tempo)
        last_tick = tick
        sample = math.floor(seconds * sample_rate + 0.5)
        if message.type == "set_tempo":
            tempo = message.tempo
            continue
        if not hasattr(message, "channel"):
            continue
        channel = (port, message.channel)
        if message.type == "program_change":
            programs[channel] = message.program
        elif message.type == "control_change" and message.control == 64:
            pedal[channel] = message.value >= 64
            if not pedal[channel]:
                for onset, pitch, velocity, program in sustained.pop(channel, []):
                    finish(onset, pitch, velocity, program, sample)
        elif message.type == "note_on" and message.velocity > 0:
            program = 128 if message.channel == 9 else programs[channel]
            active[(*channel, message.note)].append((sample, message.velocity, program))
        elif message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        ):
            key = (*channel, message.note)
            queue = active.get(key)
            if not queue:
                continue
            onset, velocity, program = queue.popleft()
            if not queue:
                del active[key]
            if program != 128 and pedal[channel]:
                sustained[channel].append((onset, message.note, velocity, program))
            else:
                finish(onset, message.note, velocity, program, sample)

    end_sample = math.floor(seconds * sample_rate + 0.5)
    for (_, _, pitch), queue in active.items():
        for onset, velocity, program in queue:
            finish(onset, pitch, velocity, program, end_sample)
    for held in sustained.values():
        for onset, pitch, velocity, program in held:
            finish(onset, pitch, velocity, program, end_sample)
    notes.sort(key=_note_key)
    length = max(end_sample, max((note.offset for note in notes), default=0))
    return MidiPerformance(tuple(notes), sample_rate, length)


def write_midi(performance: MidiPerformance, path: str | Path) -> Path:
    """Write canonical type-1 MIDI; return the saved path.

    The output has a fixed 120 BPM tempo and 24,000 ticks per beat. The
    endpoint sample clock (16 kHz) maps exactly onto its ticks. Tracks
    group notes by program. Overlapping equal pitches use separate voices so
    their intervals remain distinguishable on reread; extra voices/programs
    use standard MIDI ports when the 15 melodic channels are exhausted.
    """

    if not isinstance(performance, MidiPerformance):
        raise TypeError("performance must be MidiPerformance")
    path = Path(path)
    midi = mido.MidiFile(type=1, ticks_per_beat=24_000)

    def to_tick(sample: int) -> int:
        return math.floor(sample * 48_000 / performance.sample_rate + 0.5)

    end_tick = to_tick(performance.length_samples)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    conductor.append(mido.MetaMessage("end_of_track", time=end_tick))
    midi.tracks.append(conductor)

    # Allocate a new voice only when an equal pitch overlaps on that program.
    # This preserves nested/crossing intervals without storing the source MIDI.
    voices: dict[int, list[list[Note]]] = defaultdict(list)
    ends: dict[int, list[dict[int, int]]] = defaultdict(list)
    for note in sorted(performance.notes, key=_note_key):
        voice = next(
            (i for i, pitches in enumerate(ends[note.program])
             if pitches.get(note.pitch, 0) <= note.onset),
            len(ends[note.program]),
        )
        if voice == len(ends[note.program]):
            voices[note.program].append([])
            ends[note.program].append({})
        voices[note.program][voice].append(note)
        ends[note.program][voice][note.pitch] = note.offset

    melodic_channels = tuple(channel for channel in range(16) if channel != 9)
    melodic_voice, drum_voice = 0, 0
    for program, program_voices in sorted(voices.items()):
        for voice_notes in program_voices:
            if program == 128:
                port, channel = drum_voice, 9
                drum_voice += 1
            else:
                port = melodic_voice // len(melodic_channels)
                channel = melodic_channels[melodic_voice % len(melodic_channels)]
                melodic_voice += 1
            if port > 127:
                raise ValueError("canonical MIDI requires more than 128 MIDI ports")
            track = mido.MidiTrack()
            track.append(mido.MetaMessage("midi_port", port=port, time=0))
            if program != 128:
                track.append(mido.Message("program_change", channel=channel, program=program))
            messages = []
            for order, note in enumerate(voice_notes):
                onset_tick = to_tick(note.onset)
                offset_tick = max(onset_tick + 1, to_tick(note.offset))
                messages.append((onset_tick, 1, order, mido.Message(
                    "note_on", channel=channel, note=note.pitch, velocity=note.velocity)))
                messages.append((offset_tick, 0, order, mido.Message(
                    "note_off", channel=channel, note=note.pitch, velocity=0)))
            previous = 0
            for tick, _, _, message in sorted(messages, key=lambda event: event[:3]):
                track.append(message.copy(time=tick - previous))
                previous = tick
            track.append(mido.MetaMessage("end_of_track", time=max(0, end_tick - previous)))
            midi.tracks.append(track)
    midi.save(path)
    return path

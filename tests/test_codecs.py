# SPDX-FileCopyrightText: 2026 AnySynth contributors
# SPDX-License-Identifier: Apache-2.0

"""Semantic roundtrips, fixed layouts and deliberate rejection boundaries."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import mido
import numpy as np

from midi_span import (
    MidiPerformance, Note, read_midi, write_midi,
    encode_notes, decode_notes,
    encode_midi, decode_midi,
    canonicalize_notes,
)


def performance(notes, rate, length=None):
    return MidiPerformance(tuple(notes), rate,
                           max((n.offset for n in notes), default=0) if length is None else length)


def ordered(notes):
    return sorted(notes, key=lambda n: (n.onset, n.offset, n.program, n.pitch, n.velocity))


class MidiIOTests(unittest.TestCase):
    def test_tempo_sustain_default_program_and_drums(self):
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.extend([
            mido.MetaMessage("set_tempo", tempo=500_000),
            mido.Message("note_on", channel=0, note=60, velocity=80),
            mido.Message("control_change", channel=0, control=64, value=127, time=240),
            mido.Message("note_on", channel=0, note=60, velocity=0, time=240),
            mido.MetaMessage("set_tempo", tempo=1_000_000),
            mido.Message("note_on", channel=9, note=36, velocity=90),
            mido.Message("note_off", channel=9, note=36, time=48),
            mido.Message("control_change", channel=0, control=64, value=0, time=192),
            mido.MetaMessage("end_of_track", time=240),
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tempo.mid"
            midi.save(path)
            parsed = read_midi(path, sample_rate=16_000)
        self.assertEqual(parsed.notes, (Note(0, 16_000, 60, 80),
                                        Note(8_000, 9_600, 36, 90, 128)))
        self.assertEqual(parsed.length_samples, 24_000)

    def test_multitrack_ports_and_overlapping_equal_pitches(self):
        notes = [Note(0, 5_000, 60, 80), Note(100, 300, 60, 90)]
        notes += [Note(200, 900, 72, 50 + p % 70, p) for p in range(1, 40)]
        source = performance(notes, 48_000, 7_000)
        with tempfile.TemporaryDirectory() as directory:
            path = write_midi(source, Path(directory) / "voices.mid")
            decoded = read_midi(path, sample_rate=48_000)
        self.assertEqual(ordered(source.notes), ordered(decoded.notes))
        self.assertEqual(decoded.length_samples, source.length_samples)

    def test_empty_midi_preserves_trailing_silence(self):
        source = performance([], 16_000, 8_000)
        with tempfile.TemporaryDirectory() as directory:
            path = write_midi(source, Path(directory) / "empty.mid")
            self.assertEqual(read_midi(path, sample_rate=16_000), source)

    def test_type_two_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "async.mid"
            midi = mido.MidiFile(type=2)
            midi.tracks.append(mido.MidiTrack([mido.MetaMessage("end_of_track")]))
            midi.save(path)
            with self.assertRaises(ValueError):
                read_midi(path, sample_rate=16_000)


class MidiSpanTests(unittest.TestCase):
    def test_default_piano_layout_and_terminal_off(self):
        source = performance([Note(0, 256, 60, 90, 7)], 16_000)
        encoding = encode_notes(source)
        self.assertEqual(encoding.values.shape, (2, 288))
        self.assertEqual(encoding.values.dtype, np.float32)
        self.assertEqual(encoding.features.shape, (2, 32, 9))
        np.testing.assert_array_equal(encoding.features[0, 0, 1:3], [1, 0])
        np.testing.assert_array_equal(encoding.features[1, 16, 1:3], [1, 0])
        decoded = decode_notes(encoding)
        self.assertEqual(decoded.notes, (Note(0, 256, 60, 90),))
        self.assertEqual(decoded.length_samples, 256)

    def test_all_pitch_velocity_and_subsample_values(self):
        notes = [Note(i * 512 + i, i * 512 + i + 256, i % 128, i % 127 + 1)
                 for i in range(256)]
        source = performance(notes, 16_000)
        decoded = decode_notes(encode_notes(source))
        self.assertEqual(ordered(decoded.notes), ordered(source.notes))

    def test_pairing_is_chronological_not_by_lane(self):
        source = performance([Note(1, 900, 60, 90), Note(2, 800, 64, 70),
                              Note(900, 1000, 60, 50)], 16_000)
        decoded = decode_notes(encode_notes(source))
        self.assertEqual(ordered(decoded.notes), ordered(source.notes))

    def test_generated_piano_program_coordinates_do_not_change_program(self):
        encoding = encode_notes(performance([Note(0, 1000, 60, 70)], 16_000))
        changed = encoding.values.copy().reshape(-1, 32, 9)
        changed[..., 1:3] = [-0.5, -0.8660254]
        decoded = decode_notes(replace(encoding, values=changed.reshape(-1, 288)))
        self.assertEqual(decoded.notes[0].program, 0)

    def test_original_pgb_mode(self):
        source = performance([Note(0, 1000, 60, 80, 1), Note(0, 1000, 64, 70, 31),
                              Note(0, 1000, 48, 90, 39)], 16_000)
        decoded = decode_notes(encode_notes(source, program_mode="pgb"))
        self.assertEqual({note.program for note in decoded.notes}, {0, 24, 32})
        self.assertEqual(ordered(decoded.notes), ordered([
            replace(source.notes[0], program=0), replace(source.notes[1], program=24),
            replace(source.notes[2], program=32)]))

    def test_non_piano_drum_and_overlap_rejection(self):
        for program in (24, 128):
            with self.assertRaises(ValueError):
                encode_notes(performance([Note(0, 100, 60, 80, program)], 16_000))
        with self.assertRaises(ValueError):
            encode_notes(performance([Note(0, 100, 60, 80),
                                              Note(50, 150, 60, 90, 1)], 16_000))

    def test_explicit_reattack_target_preserves_activity_union(self):
        source = performance([
            Note(0, 1000, 60, 100), Note(0, 900, 60, 80, 1),
            Note(500, 600, 60, 40), Note(1200, 1400, 60, 60),
            Note(1400, 1600, 60, 70), Note(500, 800, 64, 90),
        ], 16_000, 2000)
        expected = performance([
            Note(0, 500, 60, 100), Note(500, 1000, 60, 100),
            Note(500, 800, 64, 90), Note(1200, 1400, 60, 60),
            Note(1400, 1600, 60, 70),
        ], 16_000, 2000)
        target = canonicalize_notes(source)
        self.assertEqual(ordered(target.notes), ordered(expected.notes))
        encoding = encode_notes(source, overlap_policy="reattack")
        self.assertEqual(decode_notes(encoding), target)

    def test_pedal_piano_file_with_explicit_reattack_policy(self):
        midi = mido.MidiFile(ticks_per_beat=8000)
        midi.tracks.append(mido.MidiTrack([
            mido.Message("control_change", control=64, value=127),
            mido.Message("note_on", note=60, velocity=100),
            mido.Message("note_off", note=60, time=500),
            mido.Message("note_on", note=60, velocity=40, time=500),
            mido.Message("note_off", note=60, time=500),
            mido.Message("control_change", control=64, value=0, time=500),
            mido.MetaMessage("end_of_track"),
        ]))
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "pedal.mid"
            midi.save(original)
            with self.assertRaises(ValueError):
                encode_midi(original)
            target = canonicalize_notes(read_midi(original, sample_rate=16_000))
            encoding = encode_midi(original, overlap_policy="reattack")
            result = decode_midi(encoding, Path(directory) / "canonical.mid")
            self.assertEqual(read_midi(result, sample_rate=16_000), target)

    def test_each_bank_has_capacity_16(self):
        notes = [Note(0, 1000, p, 80) for p in range(16)]
        self.assertEqual(len(decode_notes(
            encode_notes(performance(notes, 16_000))).notes), 16)
        with self.assertRaises(ValueError):
            encode_notes(performance(notes + [Note(0, 1000, 16, 80)], 16_000))
        # The ONs fit in separate frames while their OFFs overflow one bank.
        with self.assertRaises(ValueError):
            encode_notes(performance(
                [Note(p * 256, 10_000, p, 80) for p in range(17)], 16_000))

    def test_empty_and_wrong_clock(self):
        source = performance([], 16_000, 1000)
        self.assertEqual(decode_notes(encode_notes(source)), source)
        with self.assertRaises(ValueError):
            encode_notes(performance([], 48_000))

    def test_prefix_ties_choose_the_shorter_prefix(self):
        encoding = encode_notes(performance([], 16_000, 256))
        values = encoding.values.copy()
        values.reshape(-1, 32, 9)[..., 0] = 0.0
        self.assertEqual(decode_notes(replace(encoding, values=values)).notes, ())

    def test_unpaired_generated_endpoints_raise(self):
        encoding = encode_notes(performance([Note(0, 1000, 60, 80)], 16_000))
        for bank in (0, 1):
            values = encoding.values.copy().reshape(-1, 32, 9)
            values[:, bank * 16:(bank + 1) * 16, 0] = -1.0
            with self.assertRaises(ValueError):
                decode_notes(replace(encoding, values=values.reshape(-1, 288)))

    def test_mutated_nan_coordinates_raise(self):
        encoding = encode_notes(performance([], 16_000, 1000))
        encoding.values[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            decode_notes(encoding)

    def test_file_roundtrip(self):
        source = performance([Note(1, 513, 60, 80), Note(513, 2000, 60, 99),
                              Note(900, 1000, 64, 110)], 16_000, 3000)
        with tempfile.TemporaryDirectory() as directory:
            original = write_midi(source, Path(directory) / "input.mid")
            path = decode_midi(encode_midi(original),
                                      Path(directory) / "output.mid")
            decoded = read_midi(path, sample_rate=16_000)
        self.assertEqual(ordered(decoded.notes), ordered(source.notes))
        self.assertEqual(decoded.length_samples, source.length_samples)


if __name__ == "__main__":
    unittest.main()

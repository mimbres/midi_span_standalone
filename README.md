# MIDI Span Standalone

Encode MIDI files into two fixed musical representations and decode them back
into MIDI, without a neural model, PyTorch, or training code:

- **SpanSynth V3 SQ**: the framewise active-note conditioning used by
  **SpanSynth-edit v3-sq-v8**, with the original `fine40` instrument categories.
- **FlowAMT `v2_offset`**: physical ON/OFF endpoints from the small FlowAMT PoC.
  **Piano-only is the default**. An explicit option restores its original
  Piano/Guitar/Bass groups.

These functions reconstruct the **note content represented by the arrays**.
Instrument grouping and drum duration conventions are described below. Original
MIDI bytes, tempo maps, controller events, and track organization are not retained.

## Install

Python 3.11 or later is required. The only runtime dependencies are NumPy and Mido.

```bash
git clone https://github.com/mimbres/midi_span_standalone.git
cd midi_span_standalone
python -m pip install .
```

## MIDI file roundtrips

```python
from midi_span import (
    encode_spansynth_midi, decode_spansynth_midi,
    encode_flowamt_midi, decode_flowamt_midi,
)

# The actual v3-sq-v8 conditioning, including sustained active notes.
span = encode_spansynth_midi("arrangement.mid")
print(span.numeric.shape)  # [windows, 512, 128, 7]
decode_spansynth_midi(span, "arrangement_roundtrip.mid")

# Piano-only: a single legal program, with the original 32 x 9 endpoint layout.
amt = encode_flowamt_midi("piano.mid")
print(amt.values.shape)    # [frames, 288]
print(amt.features.shape)  # [frames, 32, 9]; a view of values
decode_flowamt_midi(amt, "piano_roundtrip.mid")

# Optional historical PoC vocabulary.
pgb = encode_flowamt_midi("piano_guitar_bass.mid", program_mode="pgb")
decode_flowamt_midi(pgb, "pgb_roundtrip.mid")
```

Each encoder returns a dataclass containing ordinary NumPy arrays and timeline
information. Each file decoder writes a canonical MIDI file and returns its
`Path`. Neither encoding stores an original MIDI file or a backup note list.

You can also work directly with sample-grid notes:

```python
from midi_span import (
    Note, MidiPerformance, encode_flowamt_notes, decode_flowamt_notes,
)

music = MidiPerformance(
    notes=(Note(onset=0, offset=16000, pitch=60, velocity=90, program=0),),
    sample_rate=16000,
    length_samples=20000,  # Include any desired trailing silence.
)
encoded = encode_flowamt_notes(music)
decoded = decode_flowamt_notes(encoded)
assert decoded == music
```

The corresponding SpanSynth functions are `encode_spansynth_notes` and
`decode_spansynth_notes`, using a 48,000 Hz sample clock. Programs are zero-based
GM numbers; internal program `128` means drums. Note intervals are half-open
`[onset, offset)` and must have positive duration.

## Representation 1: SpanSynth-edit v3-sq-v8

This is the **active-note event set**, rather than the earlier 8-coordinate
onset-slot sketch. It uses 48,000 samples/second, a 1,920-sample hop (25 Hz),
128 rows/frame, and fixed 512-frame windows (20.48 seconds). Longer files use
consecutive windows; crossing notes keep their physical IDs.

| Array | Shape | Meaning |
| --- | --- | --- |
| `kind_id` | `[W,512,128]` | Padding=0, ONSET=1, SUSTAIN=2, OFFSET=3 |
| `pitch_id` | `[W,512,128]` | Pitch + 1; padding=0 |
| `numeric` | `[W,512,128,7]` | Original float32 musical coordinates |
| `event_valid` | `[W,512,128]` | Occupied rows |
| `category_id` | `[W,512,128]` | `fine40` class; padding=-1 |
| `physical_note_id` | `[W,512,128]` | Links repeated rows of one note; padding=-1 |
| `valid_mask` | `[W,512]` | Frames inside the file timeline |

`W` is the number of windows. Numeric fields, in order, are:

| Index | Field | Coordinate |
| --- | --- | --- |
| 0 | Drum flag | Melodic=-1, drums=+1 |
| 1, 2 | Instrument category | `sin(2πc/40)`, `cos(2πc/40)` |
| 3 | Pitch | `2p/127 - 1` |
| 4 | Velocity | `2v/127 - 1` |
| 5 | Frame-local boundary | `2b/1920 - 1`; SUSTAIN uses 0 |
| 6 | Remaining duration | `2 log1p(min(d,983040)/48000) / log1p(983040/48000) - 1` |

A melodic note occupies every frame intersecting its interval. ONSET takes
precedence when a short note starts and ends within one frame; its duration
still encodes the ending. An offset exactly at a frame boundary belongs to
the preceding frame. Drums occupy only their onset frame. Rows sort by drum
flag, category, pitch, kind, and physical ID. More than 128 rows raises an
error instead of truncating notes.

The standalone decoder is new: it reconstructs boundaries and durations from
these rows. It uses `physical_note_id`, which was already present in the
original data path but is **not a model input**, to join rows and preserve
overlapping equal pitches. It requires the complete returned encoding; isolated
model input arrays or cropped sustain-only windows do not determine every onset.
For notes longer than 20.48 seconds, later rows recover the ending beyond the
initial saturated duration coordinate.

Several GM programs share a `fine40` category. Decode uses the first program
listed in each category in [spansynth.py](midi_span/spansynth.py), so raw program
identity within a category is not recovered. Programs outside that vocabulary
are rejected. Drum duration is absent from the original representation;
decoded drums use 100 ms, which can extend the returned timeline.

## Representation 2: piano-only FlowAMT endpoints

The clock is 16,000 samples/second with a 256-sample hop (62.5 Hz). Every frame
has **16 ON lanes followed by 16 OFF lanes**, each with nine float32 coordinates.
There is **no duration coordinate**. Lane membership defines endpoint type.

| Index | Field | Coordinate |
| --- | --- | --- |
| 0 | Presence | Occupied=+1, empty=-1 |
| 1, 2 | Program | Piano-only: fixed `(1,0)` |
| 3, 4, 5 | Pitch | `cos(2πp/12)`, `sin(2πp/12)`, `(p-63.5)/63.5` |
| 6, 7 | Velocity | `2v/127 - 1`, `2 log1p(v)/log(128) - 1` |
| 8 | Endpoint position inside frame | `2s/255 - 1` |

Piano-only accepts GM programs 0–7, maps them all to program 0, and rejects
other instruments and drums. Its decoder has one legal program, so changes to
generated program coordinates cannot change the output instrument. The two
program dimensions stay in the array to preserve the historical nine-coordinate
layout.

With `program_mode="pgb"`, the original groups are Piano 0–7 → 0, Guitar
24–31 → 24, and Bass 32–39 → 32. Their program coordinates are `(1,0)`,
`(-0.5,√3/2)`, and `(-0.5,-√3/2)` respectively.

Within each ON/OFF bank, endpoints form an occupied prefix ordered by sample
position, program group, pitch, velocity, then source order. More than 16
endpoints in either bank raises an error. The full-file wrapper retains the
final OFF even when it lies exactly at the file ending or a frame boundary.

Decode clamps coordinates to `[-1,1]`, selects each bank's prefix length using
the maximum cumulative presence score (including the empty prefix), then
projects payloads to the nearest legal codeword. Ties choose the lowest index.
Endpoints pair chronologically by program group and pitch, with OFF before ON
at the same sample; the ON endpoint supplies note velocity. **ON lane k does
not pair with OFF lane k.** Historical *same-index coupling* means Gaussian
lane k was coupled to clean lane k during flow training.

Overlapping notes with the same mapped program and pitch cannot be uniquely
represented by this pairing rule. The default `overlap_policy="reject"` raises
an error, including for pedal-held piano reattacks. If that input is intentional,
use the explicit `overlap_policy="reattack"` option:

```python
amt = encode_flowamt_midi("pedal_piano.mid", overlap_policy="reattack")
decode_flowamt_midi(amt, "pedal_piano_canonical.mid")
```

This option applies the historical activity-union/reattack rule **before**
encoding: equal onsets merge using the latest offset and maximum velocity,
then each connected activity interval splits at its distinct onsets. Each
resulting interval uses the maximum velocity active at its start. Gaps remain
gaps. It does not perform the original multi-source near-onset doubling merge.
The reconstructed target is the resulting nonoverlapping note sequence,
rather than each original overlapping interval. You can inspect that target
directly with `canonicalize_flowamt_notes(read_midi(path, sample_rate=16000))`.

Unlike the historical permissive materializer, this standalone decoder raises
on orphan OFFs, repeated unclosed ONs, unclosed notes, or endpoints beyond the
declared timeline. It does not silently discard endpoints.

Lane attention and mel scaling changed the model/audio frontend. Prefix CE,
rational-tail gradients, and semi-CRF changed training. They did not change
these endpoint coordinates. **Semi-CRF is a training auxiliary term and is
excluded from encode/decode.** This repository contains no model checkpoint,
flow sampler, or training loss.

## MIDI I/O and verification

The shared reader resolves tempo changes and sustain pedal (CC64), supplies
GM program 0 when no program change is present, and treats channel 9 as drums.
Times use `floor(seconds * sample_rate + 0.5)`. Release velocities, other
controllers, pitch bends, and original track/channel assignments are outside
the note representation. Type-2 and SMPTE-timed MIDI files are rejected.

The writer uses a canonical 120 BPM MIDI timeline with 24,000 ticks/beat,
which represents both sample clocks exactly. Separate voices preserve
overlapping equal-pitch SpanSynth notes on reread; standard MIDI ports allow
more than 15 melodic program/voice tracks. Trailing silence is retained except
when the SpanSynth drum-duration convention extends the ending.

Run the tests with Python's standard library:

```bash
python -m unittest discover -s tests -v
```

Tests cover MIDI file roundtrips, tempo/pedal parsing, multiple instruments and
ports, repeated notes, window crossings, long-note duration saturation, exact
frame boundaries, all legal FlowAMT pitch/velocity/subsample values, piano-only
decoding, and explicit overflow/unsupported-input errors. Extraction was also
checked directly against the original production Torch encode/decode paths.

## Origins and license

The implementation is adapted from AnySynth's
`src/anysynth/data/spansynth_v3_sq.py` (`encode_spansynth_v3_sq_midi`),
`src/anysynth/data/midi_sketch.py` (duration transforms),
`src/anysynth/data/vocabulary.py` (`fine40`), and the small FlowAMT checkout's
`src/anysynth/midi_span/amt_flow_codec_v2_offset.py`.
The PoC lineage is documented in
`docs/reports/2026-07-28-midi-span-amt-v2-v4-setattn-comparison.md` in that checkout.
The category vocabulary and sustain/FIFO parsing behavior include adaptations
from YourMT3. Source copyright notices are retained.

Released under [Apache-2.0](LICENSE).

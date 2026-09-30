# MIDI Span Standalone

Convert piano MIDI into fixed continuous ON/OFF endpoint coordinates for
**DiT conditioning and generation targets**, then decode those coordinates
back into MIDI. Encoding and decoding need **no training, neural model, or
checkpoint**. The package uses only NumPy and Mido.

The representation is adapted from the small FlowAMT `v2_offset` PoC. Piano
is the default; an explicit option retains the original Piano/Guitar/Bass
vocabulary. The output is float32 `[T,288]`: 16 ON and 16 OFF endpoints per
frame, with nine coordinates per endpoint and no learned embedding stage.

## Install

Python 3.11 or later is required.

```bash
git clone https://github.com/mimbres/midi_span_standalone.git
cd midi_span_standalone
python -m pip install .
```

## MIDI file roundtrip

```python
from midi_span import encode_flowamt_midi, decode_flowamt_midi

encoded = encode_flowamt_midi("piano.mid")
print(encoded.values.shape)    # [T,288]
print(encoded.features.shape)  # [T,32,9]; a view of values
decode_flowamt_midi(encoded, "piano_roundtrip.mid")
```

The encoder returns `FlowAMTEncoding`, containing ordinary NumPy coordinates,
`length_samples`, and the selected program mode. The decoder returns the saved
MIDI file's `Path`. No original MIDI or backup note list is stored.

Sample-grid notes are supported directly:

```python
from midi_span import (
    Note, MidiPerformance, encode_flowamt_notes, decode_flowamt_notes,
)

music = MidiPerformance(
    notes=(Note(onset=0, offset=16000, pitch=60, velocity=90, program=0),),
    sample_rate=16000,
    length_samples=20000,  # Retain trailing silence.
)
encoded = encode_flowamt_notes(music)
assert decode_flowamt_notes(encoded) == music
```

Programs use zero-based GM numbers. Note intervals are half-open
`[onset, offset)` and have positive duration. Piano programs 0–7 all decode to
program 0; other instruments and drums are rejected in the default mode.

## Encoding

1. Read MIDI tempo, note-on/off events, programs, and sustain pedal. Quantize
   event times with `floor(seconds * 16000 + 0.5)`.
2. Emit an ON at each note's onset and an OFF at its offset. For endpoint
   sample `q`, the frame is `q // 256` and its within-frame position is
   `s = q % 256`. Each frame spans **16 ms**, or 62.5 frames/second.
3. Sort each frame's ON and OFF banks independently by within-frame position,
   program group, pitch, velocity, and source order. Occupied lanes form a
   prefix; more than 16 endpoints in either bank raises an error.
4. Fill fixed float32 coordinates and flatten `[T,32,9]` to `[T,288]`.

The canvas includes the final endpoint even at the file ending or a frame
boundary: `T = length_samples // 256 + 1`. Endpoint type comes from the bank:
lanes 0–15 are ON, and lanes 16–31 are OFF. There is no duration coordinate.

| Index | Field | Coordinate |
| --- | --- | --- |
| 0 | Presence | Occupied=+1, empty=-1 |
| 1, 2 | Program | Piano-only: fixed `(1,0)` |
| 3, 4, 5 | Pitch | `cos(2πp/12)`, `sin(2πp/12)`, `(p-63.5)/63.5` |
| 6, 7 | Velocity | `2v/127 - 1`, `2 log1p(v)/log(128) - 1` |
| 8 | Position inside frame | `2s/255 - 1` |

An empty endpoint is `[-1,0,0,0,0,0,0,0,0]`. The two program coordinates
remain in the piano-only layout to preserve the original nine-coordinate
format. They do not require learned program embeddings.

## Decoding

1. Clamp the supplied coordinates to `[-1,1]`.
2. Select each bank's occupied prefix length using the maximum cumulative
   presence score, including the empty prefix. Ties choose the shortest prefix.
3. Recover pitch, velocity, and within-frame sample by nearest legal codeword.
   In piano-only mode the program is always 0.
4. Convert endpoints to absolute sample positions and pair chronologically by
   program group and pitch, processing OFF before ON at the same sample. The
   ON endpoint supplies note velocity. **ON lane k does not pair with OFF lane k.**
5. Write the resulting note intervals to canonical MIDI.

All these steps are implemented in [flowamt.py](midi_span/flowamt.py) and the
shared [MIDI I/O](midi_span/midi.py). No learned decoder is involved.

## Use with a DiT

Use `encoded.values` directly as a continuous `[T,288]` condition or clean
training target. A batch adds the leading dimension: `[B,T,288]`. Any input
projection or attention in your DiT trains with that model; the MIDI codec
itself has no parameters to learn. For variable-length batches, use the
canonical empty endpoint for padding and exclude padded frames from the loss.

After your model produces a finite float32 array, decoding requires only that
array and its timeline length:

```python
from midi_span import FlowAMTEncoding, decode_flowamt_midi

# generated_values: your model's NumPy float32 output, shape [T,288].
# length_samples: the intended output length on the 16 kHz sample clock.
generated = FlowAMTEncoding(
    values=generated_values,
    length_samples=length_samples,
)
decode_flowamt_midi(generated, "generated.mid")
```

The original MIDI and its encoding are not needed for this operation. With
flow matching, historical *same-index coupling* means Gaussian lane k is
coupled to clean lane k in training. Decode uses temporal note pairing,
regardless of lane ranks. Coordinate projection is a discrete output operation;
training can operate on the continuous coordinates before projection.

The current decoder is strict: orphan OFFs, repeated unclosed ONs, unclosed
notes at the end, and endpoints beyond `length_samples` raise errors. It
therefore does not promise a MIDI file for every arbitrary generated array.
The package supplies the fixed representation and decoder; it does not supply
a DiT, training loss, or sampling loop.

## Pedal-held reattacks

Overlapping notes sharing the same mapped program and pitch are not uniquely
recoverable with this pairing rule. By default, `overlap_policy="reject"`
raises an error. For intentional overlaps, including piano sustain-pedal
reattacks, select the explicit preprocessing option:

```python
encoded = encode_flowamt_midi("pedal_piano.mid", overlap_policy="reattack")
decode_flowamt_midi(encoded, "pedal_piano_canonical.mid")
```

This applies the historical activity-union/reattack rule before encoding:
equal onsets merge using the latest offset and maximum velocity, then each
connected activity interval splits at its distinct onsets. Each resulting
interval uses the maximum velocity active at its start. Gaps remain gaps.
The original multi-source near-onset doubling merge is not applied.

The roundtrip target is the resulting nonoverlapping note sequence. Inspect
that target with
`canonicalize_flowamt_notes(read_midi(path, sample_rate=16000))`.

## Optional historical program groups

Use `program_mode="pgb"` for the original PoC vocabulary:

- Piano 0–7 → program 0, coordinates `(1,0)`
- Guitar 24–31 → program 24, coordinates `(-0.5,√3/2)`
- Bass 32–39 → program 32, coordinates `(-0.5,-√3/2)`

Other programs and drums remain unsupported. Programs within a group share
one representative program on decode.

## MIDI I/O and verification

The reader resolves tempo changes and sustain pedal (CC64), defaults unset
programs to GM piano 0, and identifies channel 9 as drums. The endpoint codec
rejects those drums. Release velocities, other controllers, pitch bends,
original tempo maps, and track/channel assignments are not stored in the
representation. Type-2 and SMPTE-timed MIDI files are rejected.

The writer uses 120 BPM with 24,000 ticks/beat, representing the 16 kHz sample
clock exactly and retaining trailing silence. The shared MIDI I/O also
supports separate voices and standard MIDI ports.

```bash
python -m unittest discover -s tests -v
```

Tests cover MIDI file roundtrips, tempo/pedal parsing, program mapping, note
reattacks, exact sample/frame boundaries, all legal pitch/velocity/subsample
values, piano-only projection, prefix ties, and explicit overflow/invalid
endpoint errors. The coordinates and projections were also compared directly
with the original production Torch implementation.

## Origins and license

The fixed coordinates and projection rules are adapted from the small FlowAMT
checkout's `src/anysynth/midi_span/amt_flow_codec_v2_offset.py`. The historical
lineage is documented in
`docs/reports/2026-07-28-midi-span-amt-v2-v4-setattn-comparison.md` in that checkout.
The sustain/FIFO parsing behavior includes adaptations from YourMT3. Source
copyright notices are retained.

Lane attention and mel scaling changed the original model/audio frontend.
Prefix CE, rational-tail gradients, and semi-CRF changed training, without
changing these endpoint coordinates. **Semi-CRF is a training auxiliary term
and is excluded from encode/decode.**

Released under [Apache-2.0](LICENSE).

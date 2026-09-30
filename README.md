# MIDI Span Standalone

Convert piano MIDI into fixed continuous ON/OFF endpoint coordinates for
**DiT conditioning and generation targets**, then decode those coordinates
back into MIDI. Encoding and decoding need **no training, neural model, or
checkpoint**. The fixed codec uses only NumPy and Mido. Optional PyTorch
functions provide field-balanced MSE, differentiable ODE exposure, and
decoder-aligned Prefix CE with rational-tail gradients for your model.

The representation is adapted from the small FlowAMT `v2_offset` PoC. Piano
is the default; an explicit option retains the original Piano/Guitar/Bass
vocabulary. The output is float32 `[T,288]`: 16 ON and 16 OFF endpoints per
frame, with nine coordinates per endpoint and no learned embedding stage.

## Install

Python 3.11 or later is required.

```bash
git clone https://github.com/mimbres/midi_span_standalone.git
cd midi_span_standalone
python -m pip install -e .
```

To include the training functions:

```bash
python -m pip install -e '.[training]'
```

The distribution name is `midi-span-standalone`, and the import package is
`midi_span`. The source AnySynth projects use `anysynth.midi_span`, so these
packages can coexist in one environment. This package does not install or
replace `anysynth`. Two AnySynth clones still share their own `anysynth`
distribution and should not both be editable installs in one environment.

## MIDI file roundtrip

```python
from midi_span import encode_midi, decode_midi

encoded = encode_midi("piano.mid")
print(encoded.values.shape)    # [T,288]
print(encoded.features.shape)  # [T,32,9]; a view of values
decode_midi(encoded, "piano_roundtrip.mid")
```

The encoder returns `MidiSpanEncoding`, containing ordinary NumPy coordinates,
`length_samples`, and the selected program mode. The decoder returns the saved
MIDI file's `Path`. No original MIDI or backup note list is stored.

Sample-grid notes are supported directly:

```python
from midi_span import (
    Note, MidiPerformance, encode_notes, decode_notes,
)

music = MidiPerformance(
    notes=(Note(onset=0, offset=16000, pitch=60, velocity=90, program=0),),
    sample_rate=16000,
    length_samples=20000,  # Retain trailing silence.
)
encoded = encode_notes(music)
assert decode_notes(encoded) == music
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
from midi_span import MidiSpanEncoding, decode_midi

# generated_values: your model's NumPy float32 output, shape [T,288].
# length_samples: the intended output length on the 16 kHz sample clock.
generated = MidiSpanEncoding(
    values=generated_values,
    length_samples=length_samples,
)
decode_midi(generated, "generated.mid")
```

The original MIDI and its encoding are not needed for this operation. With
flow matching, historical *same-index coupling* means Gaussian lane k is
coupled to clean lane k in training. Decode uses temporal note pairing,
regardless of lane ranks. Coordinate projection is a discrete output operation;
training can operate on the continuous coordinates before projection.

The current decoder is strict: orphan OFFs, repeated unclosed ONs, unclosed
notes at the end, and endpoints beyond `length_samples` raise errors. It
therefore does not promise a MIDI file for every arbitrary generated array.
The package supplies the fixed representation, decoder, and optional training
functions below. Your project supplies the DiT and its conditioning frontend.

## Training functions

Import [training.py](midi_span/training.py) explicitly. Normal `import midi_span`
does not import PyTorch. All training functions accept either `[B,T,288]` or
`[B,T,32,9]`; tensors within one call must use the same layout and device.
`valid_mask` is an optional boolean `[B,T]` mask, with at least one valid frame
per example.

| Function | Purpose |
| --- | --- |
| `field_mse(prediction, target, clean, valid_mask=...)` | Balance the five musical fields using clean endpoint occupancy. Returns `FieldLoss` with `.total` and each field's MSE. |
| `differentiable_euler(velocity_fn, noise, steps=4, valid_mask=...)` | Generate an endpoint while retaining gradients through the Euler steps. The callback binds your model's conditions and padding mask. |
| `prefix_cross_entropy(endpoint, clean, valid_mask=...)` | Supervise ON/OFF endpoint counts with decoder-aligned Prefix CE and rational-tail backward gradients. Returns `PrefixLoss` with `.total` and group means/counts. |
| `prefix_logits(endpoint)` | Inspect the 17 count logits for each ON/OFF bank, shaped `[B,T,2,17]`. Their argmax matches the decoder's prefix rule. |

`field_mse` averages ON occupied/empty and OFF occupied/empty presence errors
equally over available groups. Program, pitch, velocity, and within-frame time
are supervised only on occupied lanes. Program loss balances the available
P/G/B groups, including the single constant piano group for piano-only input.
The total averages available fields, rather than all 288 coordinates.

Prefix CE uses the clean endpoint counts 0–16 as labels. Logits are four times
the cumulative hard-clamped presence scores, including the empty prefix. CE
is normalized by `log(17)` and balances ON-empty/nonempty and OFF-empty/nonempty
frame groups. Its rational-tail backward gradient is one inside `[-1,1]` and
`1/(1+d)^2` outside, where `d` is distance from that interval. Forward projection
stays identical to the decoder. A perfect clean endpoint still has positive
soft CE. Neither count supervision nor the field loss guarantees valid
chronological ON/OFF pairing for every generated sample.

### Connect to your velocity model

This example assumes FP32 `clean` endpoints `[B,T,288]`, a tensor `condition`
with the same batch dimension, a boolean `valid_mask`, and your velocity model
with the illustrated call signature. Adapt the calls to your model's API.

```python
import torch
from midi_span.training import (
    field_mse, differentiable_euler, prefix_cross_entropy,
)

# Same-index linear flow matching: no endpoint assignment or matching stage.
noise = torch.randn_like(clean).masked_fill(~valid_mask[..., None], 0.0)
t = torch.rand(clean.shape[0], device=clean.device, dtype=torch.float32)
state = (1 - t[:, None, None]) * noise + t[:, None, None] * clean
state = state.masked_fill(~valid_mask[..., None], 0.0)
prediction = velocity_model(state, t, condition, valid_mask=valid_mask)
flow = field_mse(prediction, clean - noise, clean, valid_mask=valid_mask)

# Reuse CFM noise on a smaller batch for the extra four model evaluations.
k = min(16, clean.shape[0])
endpoint_mask = valid_mask[:k]

def exposure_velocity(state, t):
    return velocity_model(state, t, condition[:k], valid_mask=endpoint_mask)

endpoint = differentiable_euler(
    exposure_velocity, noise[:k], steps=4, valid_mask=endpoint_mask,
)
exposure = field_mse(endpoint, clean[:k], clean[:k], valid_mask=endpoint_mask)
prefix = prefix_cross_entropy(endpoint, clean[:k], valid_mask=endpoint_mask)
loss = flow.total + 0.1 * exposure.total + 0.1 * prefix.total
loss.backward()
```

Both endpoint losses share the same rollout. Prefix CE adds no model call.
ODE exposure keeps the four model-call graphs for backward, so the subset
size controls its memory cost. All loss reductions and state updates use FP32;
prefix accumulation uses FP64 as in the fixed decoder and requires a device
that supports FP64, such as CPU or CUDA.

The example weights and four-step/16-example exposure reflect the historical
PoC continuation. They are not a validated fresh piano-only training recipe.
Your trainer controls loss weights, their warmup schedules, subset size, and
optimizer. These functions require no separate encoder/decoder training.
For sampling, use `differentiable_euler` under `torch.no_grad()` with your chosen
step count, then convert its output to NumPy for `decode_midi`.

## Pedal-held reattacks

Overlapping notes sharing the same mapped program and pitch are not uniquely
recoverable with this pairing rule. By default, `overlap_policy="reject"`
raises an error. For intentional overlaps, including piano sustain-pedal
reattacks, select the explicit preprocessing option:

```python
encoded = encode_midi("pedal_piano.mid", overlap_policy="reattack")
decode_midi(encoded, "pedal_piano_canonical.mid")
```

This applies the historical activity-union/reattack rule before encoding:
equal onsets merge using the latest offset and maximum velocity, then each
connected activity interval splits at its distinct onsets. Each resulting
interval uses the maximum velocity active at its start. Gaps remain gaps.
The original multi-source near-onset doubling merge is not applied.

The roundtrip target is the resulting nonoverlapping note sequence. Inspect
that target with
`canonicalize_notes(read_midi(path, sample_rate=16000))`.

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
With the training extra, tests also cover field/group balance, padding and null
payload exclusion, rational-tail gradients, and gradient flow through Euler
exposure. Losses and gradients were compared with the original PoC functions.

## Origins and license

The fixed coordinates and projection rules are adapted from the small FlowAMT
checkout's `src/anysynth/midi_span/amt_flow_codec_v2_offset.py`. The historical
lineage is documented in
`docs/reports/2026-07-28-midi-span-amt-v2-v4-setattn-comparison.md` in that checkout.
The sustain/FIFO parsing behavior includes adaptations from YourMT3. Source
copyright notices are retained.

Lane attention and mel scaling changed the original model/audio frontend.
This package includes the field loss, ODE exposure, and rational-tail Prefix CE
without importing that model or frontend. The later semi-CRF needed an extra
training head and interval targets and is excluded from this package. None of
these training additions change the endpoint coordinates or fixed decoder.

Released under [Apache-2.0](LICENSE).

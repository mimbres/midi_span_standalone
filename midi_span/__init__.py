"""Two fixed MIDI span representations, with no model or training dependency."""

from .midi import MidiPerformance, Note, read_midi, write_midi
from .spansynth import (
    SpanSynthEncoding,
    encode_spansynth_midi,
    encode_spansynth_notes,
    decode_spansynth_midi,
    decode_spansynth_notes,
)
from .flowamt import (
    FlowAMTEncoding,
    canonicalize_flowamt_notes,
    encode_flowamt_midi,
    encode_flowamt_notes,
    decode_flowamt_midi,
    decode_flowamt_notes,
)

__all__ = [
    "MidiPerformance", "Note", "read_midi", "write_midi",
    "SpanSynthEncoding", "encode_spansynth_midi", "encode_spansynth_notes",
    "decode_spansynth_midi", "decode_spansynth_notes",
    "FlowAMTEncoding", "canonicalize_flowamt_notes",
    "encode_flowamt_midi", "encode_flowamt_notes",
    "decode_flowamt_midi", "decode_flowamt_notes",
]

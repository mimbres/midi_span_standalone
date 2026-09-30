"""Fixed FlowAMT MIDI endpoint encoding, with no model or training dependency."""

from .midi import MidiPerformance, Note, read_midi, write_midi
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
    "FlowAMTEncoding", "canonicalize_flowamt_notes",
    "encode_flowamt_midi", "encode_flowamt_notes",
    "decode_flowamt_midi", "decode_flowamt_notes",
]

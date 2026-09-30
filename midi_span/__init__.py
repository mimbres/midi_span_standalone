"""Fixed MIDI endpoint encoding, with no model or training dependency."""

from .midi import MidiPerformance, Note, read_midi, write_midi
from .flowamt import (
    MidiSpanEncoding,
    canonicalize_notes,
    encode_midi,
    encode_notes,
    decode_midi,
    decode_notes,
)

__all__ = [
    "MidiPerformance", "Note", "read_midi", "write_midi",
    "MidiSpanEncoding", "canonicalize_notes",
    "encode_midi", "encode_notes",
    "decode_midi", "decode_notes",
]

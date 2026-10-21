from pydub import AudioSegment
from pydub.generators import Sine

from meetingsgpt.audio import _merge_parts


def _tone(ms: int, freq: int = 440) -> AudioSegment:
    return Sine(freq).to_audio_segment(duration=ms).set_frame_rate(16000).set_channels(1)


def test_parts_fitting_exactly_max_are_merged_into_one_chunk():
    chunks = _merge_parts([_tone(400), _tone(600)], max_ms=1000, overlap_ms=0)
    assert [len(c) for c in chunks] == [1000]


def test_part_crossing_max_starts_a_new_chunk():
    chunks = _merge_parts([_tone(400), _tone(400), _tone(400)], max_ms=1000, overlap_ms=0)
    assert [len(c) for c in chunks] == [800, 400]


def test_part_longer_than_max_is_cut_at_fixed_duration():
    chunks = _merge_parts([_tone(2500)], max_ms=1000, overlap_ms=0)
    assert [len(c) for c in chunks] == [1000, 1000, 500]


def test_long_part_windows_overlap_so_cut_words_are_not_lost():
    chunks = _merge_parts([_tone(2500)], max_ms=1000, overlap_ms=100)
    assert [len(c) for c in chunks] == [1000, 1000, 700]


def test_audio_is_preserved_in_order():
    parts = [_tone(300, 220), _tone(1700, 440), _tone(500, 880)]
    chunks = _merge_parts(parts, max_ms=1000, overlap_ms=0)
    assert all(len(c) <= 1000 for c in chunks)
    assert b"".join(c.raw_data for c in chunks) == b"".join(p.raw_data for p in parts)


def test_no_parts_yields_no_chunks():
    assert _merge_parts([], max_ms=1000, overlap_ms=0) == []

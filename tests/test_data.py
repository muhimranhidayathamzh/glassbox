import json

import numpy as np

from glassbox.data import TRAIN_HEAD_NAME, VALID_FILE, encode_stories, prepare, split_stories
from glassbox.tokenizer import EOT, BPETokenizer


def test_split_stories_drops_truncated_tail_and_non_ascii():
    text = f"Story one.\n{EOT}\n  Story two.  \n{EOT}\nCafé story.\n{EOT}\nTruncated sto"
    stories, stats = split_stories(text)
    assert stories == ["Story one.", "Story two."]
    assert stats == {"segments": 4, "dropped_truncated_tail": 1, "dropped_non_ascii": 1, "kept": 2}


def test_split_stories_keeps_last_story_when_file_ends_with_marker():
    stories, stats = split_stories(f"A.\n{EOT}\nB.\n{EOT}\n")
    assert stories == ["A.", "B."]
    assert stats["dropped_truncated_tail"] == 0


def test_encode_stories_appends_eot_after_each_story():
    tok = BPETokenizer.train("the cat sat on the mat " * 10, vocab_size=265)
    ids = encode_stories(tok, ["the cat", "a mat"])
    assert ids.dtype == np.uint16
    assert (ids == tok.eot_id).sum() == 2 and ids[-1] == tok.eot_id
    assert tok.decode(ids.tolist()) == f"the cat{EOT}a mat{EOT}"


def test_prepare_end_to_end(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    stories = [f"Once upon a time, a dog named Rex{i} ran home." for i in range(50)]
    (raw / TRAIN_HEAD_NAME).write_text(f"\n{EOT}\n".join(stories) + f"\n{EOT}\nPartial st", encoding="utf-8")
    (raw / VALID_FILE).write_text(f"\n{EOT}\n".join(stories[:5]) + f"\n{EOT}\n", encoding="utf-8")

    stats = prepare(raw, out, vocab_size=280)

    tok = BPETokenizer.load(out / "tokenizer.json")
    train_ids = np.fromfile(out / "train.bin", dtype=np.uint16)
    assert tok.decode(train_ids.tolist()) == "".join(s + EOT for s in stories)
    assert stats["train"]["kept"] == 50 and stats["train"]["dropped_truncated_tail"] == 1
    assert stats["val"]["kept"] == 5 and stats["val"]["dropped_truncated_tail"] == 0
    assert json.loads((out / "data_stats.json").read_text())["vocab_size"] == 280

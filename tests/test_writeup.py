import json
import re
from pathlib import Path

import pytest

from glassbox.eval.writeup import num_id, outputs

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
needs_results = pytest.mark.skipif(not (RESULTS / "guard_layers.json").exists(), reason="run the experiments first")


def result(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def test_indonesian_number_punctuation():
    assert num_id("2,048") == "2.048"
    assert num_id("2.6") == "2,6"
    assert num_id("1,234.5") == "1.234,5"


@needs_results
def test_texts_quote_the_result_files(tmp_path):
    files = outputs(RESULTS, tmp_path / "web", tmp_path / "docs")
    page = files[tmp_path / "web" / "case-study.html"]
    post_en = files[tmp_path / "docs" / "linkedin_post_en.md"]
    post_id = files[tmp_path / "docs" / "linkedin_post_id.md"]

    best = max(result("steering")["summary"]["by_alpha"], key=lambda r: r["feature_mention"])
    guard = {m["name"]: m["new"] for m in result("guard")["methods"]}
    for text in (page, post_en, post_id):
        assert f"{best['feature_mention'] * 100:.0f}%" in text and f"{best['random_mention'] * 100:.0f}%" in text
        assert f"{guard['regex or SAE guard']['recall'] * 100:.0f}%" in text
        assert f"{guard['regex + name list']['recall'] * 100:.0f}%" in text
        assert "Claude Code" in text
    assert f"{result('web_parity')['js_vs_torch']['max_abs_diff']:.1e}" in page
    assert f"{result('lm_eval')['n_params'] / 1e6:.1f}M" in post_en
    assert num_id(f"{result('lm_eval')['n_params'] / 1e6:.1f}") + " juta" in post_id


@needs_results
def test_page_has_no_unfilled_template_fields_and_valid_links(tmp_path):
    page = outputs(RESULTS, tmp_path / "web", tmp_path / "docs")[tmp_path / "web" / "case-study.html"]
    assert "{" not in page and "}" not in page
    for href in re.findall(r'(?:href|src)="([^"]+)"', page):
        if not href.startswith("http"):
            assert (ROOT / "web" / href).exists(), href


@needs_results
def test_shipped_texts_are_up_to_date():
    for path, text in outputs(RESULTS, ROOT / "web", ROOT / "docs").items():
        assert path.read_text(encoding="utf-8") == text, f"{path} is stale: run python -m glassbox.eval.writeup"

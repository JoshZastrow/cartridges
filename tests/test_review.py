import json

from cartridges.training.review import build_review, parse_flags, word_diff_html


def test_word_diff_marks_missing_and_extra_words_and_escapes_html():
    assert word_diff_html("a <b> c", "a <b> d") == "a &lt;b&gt; <del>c</del><ins>d</ins>"
    assert word_diff_html("one two", "one") == "one<del> two</del>"


def test_parse_flags_keeps_names_only():
    assert parse_flags("tall_image(5201px, 2.9x median); over_1024_tokens(1356)") == ["over_1024_tokens", "tall_image"]
    assert parse_flags("") == []


def test_build_review_joins_runs_and_metadata(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    row = {"split": "test", "index": 0, "journal": "J", "page": 3, "reference": "# Title\nhello", "cer": 0.5, "status": "ok"}
    (results / "a.json").write_text(json.dumps([{**row, "hypothesis": "hello", "looped": False}]))
    (results / "b.json").write_text(json.dumps([{**row, "hypothesis": None, "status": "error"}]))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"split": "test", "index": 0, "set": "loop"}) + "\n")
    quality = tmp_path / "pages.csv"
    quality.write_text('journal,page,flags\nJ,3,"tall_image(9px, 3x median)"\n')

    out = tmp_path / "review.html"
    summary = build_review(results, out, manifest=manifest, quality_csv=quality)
    assert summary == {"out": str(out), "runs": ["a", "b"], "n_pages": 1}
    data = json.loads(out.read_text().split("const DATA = ", 1)[1].split(";\n", 1)[0])
    page = data["pages"][0]
    assert page["set"] == "loop" and page["flags"] == ["tall_image"] and page["heading_label"]
    assert page["runs"]["b"]["cer"] is None and page["runs"]["a"]["ned"] > 0

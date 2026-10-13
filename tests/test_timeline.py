import json

from cartridges.batch.timeline import build_timeline, read_entries


def write_transcript(out, journal, pages):
    """pages: [(page number, date label, text)] in the transcript.md format."""
    d = out / journal
    d.mkdir(parents=True)
    sections = [f"## p. {n} - date: {label}\n\n{text}\n" for n, label, text in pages]
    (d / "transcript.md").write_text("\n---\n".join(s.rstrip() for s in sections) + "\n")


def test_read_entries_groups_continuation_pages(tmp_path):
    write_transcript(tmp_path, "A", [
        (1, "2021-11-19 (est.)", "# Title"),
        (2, "2021-11-19", "Nov 19, 2021\nfirst"),
        (3, "2021-11-19 (cont.)", "more"),
        (4, "2021-12-05 (?)", "Dec 5\nnext"),
    ])
    entries = read_entries(tmp_path / "A" / "transcript.md")
    assert [(e.date, e.marker, [n for n, _ in e.pages]) for e in entries] == [
        ("2021-11-19", "(est.)", [1]),
        ("2021-11-19", "", [2, 3]),
        ("2021-12-05", "(?)", [4]),
    ]
    assert entries[1].pages[1] == (3, "more")
    assert entries[1].title == "2021-11-19 · A p. 2–3"


def test_build_timeline_orders_entries_across_journals(tmp_path):
    write_transcript(tmp_path, "Later", [(1, "2024-10-21", "Oct 21, 2024\n## a heading in the text")])
    write_transcript(tmp_path, "Earlier", [
        (1, "2021-11-19", "Nov 19, 2021"),
        (2, "2024-10-21", "Oct 21, 2024\nsame day"),
    ])
    summary = build_timeline(tmp_path)
    assert summary["journals"] == ["Earlier", "Later"]
    assert (summary["entries"], summary["pages"]) == (3, 3)

    markdown = (tmp_path / "timeline.md").read_text()
    titles = [line for line in markdown.splitlines() if line.startswith("#")]
    assert titles == [
        "# Journals",
        "## 2021", "### 2021-11", "#### 2021-11-19 · Earlier p. 1",
        "## 2024", "### 2024-10", "#### 2024-10-21 · Earlier p. 2", "#### 2024-10-21 · Later p. 1",
    ]
    assert "\\## a heading in the text" in markdown  # page headings are escaped

    tree = json.loads((tmp_path / "timeline_pageindex_tree.json").read_text())
    root = tree["structure"][0]
    year_2024 = root["nodes"][1]
    assert (root["node_id"], root["start_index"], root["end_index"]) == ("0000", 1, 3)
    assert (year_2024["title"], year_2024["start_index"], year_2024["end_index"]) == ("2024", 2, 3)
    leaf = year_2024["nodes"][0]["nodes"][1]
    assert markdown.splitlines()[leaf["line_num"] - 1] == "#### 2024-10-21 · Later p. 1"


def test_each_journal_keeps_its_page_order(tmp_path):
    # p. 2 is dated a few days before p. 1 (written out of order): it stays after it.
    write_transcript(tmp_path, "A", [(1, "2021-12-21", "x"), (2, "2021-12-05", "y"), (3, "2022-01-06", "z")])
    write_transcript(tmp_path, "B", [(1, "2021-12-30", "w")])
    build_timeline(tmp_path)
    titles = [l for l in (tmp_path / "timeline.md").read_text().splitlines() if l.startswith("#### ")]
    assert titles == [
        "#### 2021-12-21 · A p. 1", "#### 2021-12-05 · A p. 2",
        "#### 2021-12-30 · B p. 1", "#### 2022-01-06 · A p. 3",
    ]

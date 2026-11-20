import json

from pile_scan import extract


def test_prompt_lists_questions():
    p = extract.build_prompt({"author": "Who wrote it?"})
    assert '"author": Who wrote it?' in p
    assert "<meta>" in p and "<text>" in p


def test_parse_well_formed():
    meta = {"doc_title": "Lease", "page_number_printed": "Page 2 of 5", "dates": ["2001-02-03"],
            "starts_mid_sentence": True, "ends_mid_sentence": False, "answers": {"author": "Ann"}}
    raw = f"<|channel>thought\n<channel|><meta>{json.dumps(meta)}</meta>\n<text>\nTitle\n\nbody\nend.\n</text><turn|>"
    v = extract.parse_response(raw)
    assert v["doc_title"] == "Lease"
    assert (v["page_number_int"], v["page_total_int"]) == (2, 5)
    assert v["ocr_text"] == "Title\n\nbody\nend."
    assert (v["first_line"], v["last_line"]) == ("Title", "end.")
    assert v["starts_mid_sentence"] == 1 and v["ends_mid_sentence"] == 0
    assert json.loads(v["answers_json"]) == {"author": "Ann"}
    assert v["truncated"] == 0
    assert v["raw_response"] == raw


def test_parse_truncated_text():
    v = extract.parse_response('<meta>{"doc_title": "X"}</meta>\n<text>\nline one\nline tw')
    assert v["truncated"] == 1
    assert v["ocr_text"] == "line one\nline tw"
    assert v["doc_title"] == "X"


def test_parse_bad_json_and_code_fence():
    v = extract.parse_response('<meta>```json\n{"doc_title": "Y", "dates": ["1999"],}\n```</meta><text>hi</text>')
    assert v["doc_title"] == "Y"
    assert json.loads(v["dates_json"]) == ["1999"]


def test_parse_ignores_format_entirely():
    v = extract.parse_response("Just some text the model wrote.")
    assert v["ocr_text"] == "Just some text the model wrote."
    assert v["doc_title"] is None
    assert v["meta_json"] is None


def test_angle_brackets_in_page_text_survive():
    v = extract.parse_response("<meta>{}</meta><text>a <b> c <illegible></text>")
    assert v["ocr_text"] == "a <b> c <illegible>"


def test_page_numbers():
    assert extract.parse_page_number("12") == (12, None)
    assert extract.parse_page_number("3/7") == (3, 7)
    assert extract.parse_page_number("Page iv") == (4, None)
    assert extract.parse_page_number("xii") == (12, None)
    assert extract.parse_page_number(None) == (None, None)
    assert extract.parse_page_number("Appendix") == (None, None)


def test_null_strings_become_none():
    v = extract.parse_response('<meta>{"doc_title": "null", "language": "N/A"}</meta><text>x</text>')
    assert v["doc_title"] is None and v["language"] is None

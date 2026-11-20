# pile-scan

Scan, sort and index a backlog of document images privately, using open models on Google Colab's GPUs.

The plan is three notebooks:

1. **Scan** (`notebooks/01_scan.ipynb`, this release): read every page with Gemma 4 12B, store the transcription and ordering clues in SQLite.
2. **Embed** (planned): add page embeddings with sqlite-vec, mainly to help group pages that belong together.
3. **Sort** (planned): use a bigger model to put pages in order within each group.

No hosted model APIs are used. Pages are read by a model running on the notebook's own GPU.

## Running the scanner

1. Put your scans in a Drive folder, e.g. `My Drive/pile-scan/inbox`. Images (PNG, JPEG, TIFF including multi-page, WebP, BMP, GIF) and PDFs are read; anything else is skipped and listed. Subfolders are included.
2. Open [`notebooks/01_scan.ipynb` in Colab](https://colab.research.google.com/github/daverooney/pile-scan/blob/main/notebooks/01_scan.ipynb).
3. **Runtime > Change runtime type**: T4 GPU (free) or L4 (Pro).
4. Fill in the settings cell, then **Runtime > Run all**. Allow Drive access when asked.

The database is written to `OUTPUT_DIR/DB_NAME` on Drive.

**When Colab disconnects**, reconnect and Run all again. The notebook copies the database back from Drive and skips every page already done. You lose at most the pages since the last checkpoint (`CHECKPOINT_EVERY`, default 10).

If this repo is private, add a Colab secret named `GITHUB_TOKEN` with a read-only token so the install cell can fetch the code.

### Settings worth knowing

| Setting | Default | Notes |
|---|---|---|
| `QUANTIZATION` | `4bit` | `4bit` fits a free T4. `8bit` needs roughly 14 GB for weights, so use an L4. |
| `MAX_SOFT_TOKENS` | `1120` | Gemma 4's visual token budget: 70, 140, 280, 560 or 1120. Transformers defaults to 280; 1120 keeps the most detail for OCR. |
| `MAX_NEW_TOKENS` | `3072` | Pages that hit this are flagged `truncated = 1`. |
| `SAMPLE_LIMIT` | `0` (all) | Stop after N pages. Files are visited in sorted order, so N always means the same pages. |
| `RETRY_ERRORS` | off | Re-read pages that failed on an earlier run. |
| `DISCONNECT_WHEN_DONE` | off | Releases the GPU at the end so a finished run stops spending compute units. |

The first scanned page logs how many image tokens the model actually received. It should be a little under the budget (around 1080 for 1120). A number near 280 means the setting didn't take effect on that Transformers version.

### Comparing 4-bit with 8-bit

Run the same 20 pages twice into different databases (`SAMPLE_LIMIT = 20`, `DB_NAME = sample_4bit.db`, then `8bit` on an L4 into `sample_8bit.db`), then run the compare cell at the bottom of the notebook. It reports text similarity per page and whether the numbers (dates, amounts, IDs) match, since that's where lower precision tends to slip.

## What's in the database

`pages` has one row per page, keyed by `(file_sha256, page_index)`:

- `ocr_text`: the transcription. `pdf_text_layer` holds a PDF's embedded text when it has one.
- Ordering clues for the sort pass: `doc_title`, `doc_type`, `page_number_printed` (plus parsed `page_number_int` and `page_total_int` from "3 of 7"), `dates_json`, `primary_date`, `first_line`, `last_line`, `running_header`, `running_footer`, `starts_mid_sentence`, `ends_mid_sentence`, `language`.
- `answers_json`: answers to the questions in the settings cell.
- Provenance: `raw_response`, `model_id`, `quantization`, `max_soft_tokens`, `image_soft_tokens`, `prompt_version`, `run_id`, timings.

`files` has one row per distinct file content (by SHA-256), so renamed or duplicated files are read once. `file_paths` lists every path each file was seen at. `runs` records each run's settings.

`pages.id` is a stable integer, meant as the key for the embeddings table (a sqlite-vec virtual table with `rowid = pages.id`) and for the sort pass's group and order tables.

## How resuming works

- The working database lives on the VM's local disk, because SQLite on a mounted Drive is unreliable.
- At start, the newest healthy copy among Drive's `DB_NAME` and `DB_NAME.prev` is copied to local disk.
- Every `CHECKPOINT_EVERY` pages, a consistent snapshot is copied to Drive under a temporary name, integrity-checked, then renamed into place. The previous copy is kept as `.prev`, so a kill at any moment leaves at least one good copy.
- Files are identified by content hash. Hashes are cached by path, size and modification time, so restarts don't re-read unchanged files from Drive.

Drive syncs in the background, so a checkpoint written just before Colab kills the VM may not have reached Google's servers. The `.prev` copy covers that.

## Development

```
pip install -e ".[test]"
pytest
```

The tests cover discovery, hashing, PDF and TIFF page splitting, reply parsing, checkpoints and resume after a simulated disconnect, using a fake model. `pile_scan/model.py` (the Gemma loader) needs a GPU and is only exercised in Colab.

# pile-scan

Scan, sort and index a backlog of document images privately, using open models on Google Colab's GPUs.

The plan is three notebooks:

1. **Scan** (`notebooks/01_scan.ipynb`, this release): read every page with Gemma 4 12B in llama.cpp, store the transcription and ordering clues in SQLite.
2. **Embed** (planned): add page embeddings with sqlite-vec, mainly to help group pages that belong together.
3. **Sort** (planned): use a bigger model to put pages in order within each group.

No hosted model APIs are used. Pages are read by llama.cpp's `llama-server` running on the notebook's own GPU, with GGUF builds of Gemma 4 12B from [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF).

## Running the scanner

1. Put your scans in a Drive folder, e.g. `My Drive/pile-scan/inbox`. Images (PNG, JPEG, TIFF including multi-page, WebP, BMP, GIF) and PDFs are read; anything else is skipped and listed. Subfolders are included.
2. Open [`notebooks/01_scan.ipynb` in Colab](https://colab.research.google.com/github/daverooney/pile-scan/blob/main/notebooks/01_scan.ipynb).
3. **Runtime > Change runtime type**: T4 GPU (free) or L4 (Pro).
4. Fill in the settings cell, then **Runtime > Run all**. Allow Drive access when asked.

The database is written to `OUTPUT_DIR/DB_NAME` on Drive.

**The first session on each GPU type compiles llama.cpp** (a T4 and an L4 each need their own build). The binary is saved to `CACHE_DIR` on Drive, so later sessions copy it back in seconds. Tick `REBUILD_LLAMA_SERVER` to force a fresh build. The model files (7.4 GB for 4-bit, 12.7 GB for 8-bit) download each session unless you tick `CACHE_MODELS_ON_DRIVE`.

**When Colab disconnects**, reconnect and Run all again. The notebook copies the database back from Drive and skips every page already done. You lose at most the pages since the last checkpoint (`CHECKPOINT_EVERY`, default 10).

If this repo is private, add a Colab secret named `GITHUB_TOKEN` with a read-only token so the install cell can fetch the code.

### Settings worth knowing

| Setting | Default | Notes |
|---|---|---|
| `QUANTIZATION` | `4bit` | `4bit` is `gemma-4-12b-it-UD-Q4_K_XL.gguf` (7.4 GB), fits a free T4. `8bit` is `gemma-4-12b-it-Q8_0.gguf` (12.7 GB), use an L4. |
| `MAX_SOFT_TOKENS` | `1120` | Gemma 4's visual token budget: 70, 140, 280, 560 or 1120. Passed to llama-server as `--image-min-tokens` and `--image-max-tokens`, with batch and ubatch at 2048 so an image's tokens fit in one pass. |
| `MAX_NEW_TOKENS` | `3072` | Pages that hit this are flagged `truncated = 1`. |
| `SAMPLE_LIMIT` | `0` (all) | Stop after N pages. Files are visited in sorted order, so N always means the same pages. |
| `RETRY_ERRORS` | off | Re-read pages that failed on an earlier run. |
| `CACHE_MODELS_ON_DRIVE` | off | Keep the GGUF files on Drive too. Costs Drive space, saves a download per session. |
| `DISCONNECT_WHEN_DONE` | off | Releases the GPU at the end so a finished run stops spending compute units. |

The llama.cpp version is pinned in `pile_scan/config.py` (`LLAMA_CPP_REF`, currently `v1.0.0`). Changing it triggers a new build.

The first scanned page logs how many image tokens the model actually received (worked out from the server's prompt token counts). It should be close to the budget. A number near 280 means the setting didn't take effect.

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

Dependencies are managed with [uv](https://docs.astral.sh/uv/). `uv.lock` is the lockfile; `requirements-colab.txt` is exported from it for the notebook, which installs into Colab's own Python.

```
uv sync
uv run pytest
```

After changing dependencies in `pyproject.toml`:

```
uv lock
uv export --frozen --no-dev --no-emit-project --no-hashes -o requirements-colab.txt
```

The tests cover discovery, hashing, PDF and TIFF page splitting, reply parsing, checkpoints, resume after a simulated disconnect, the llama.cpp build cache, model downloads, and the HTTP calls to llama-server (against a fake server in `tests/fake_llama_server.py`). The real CUDA build and Gemma 4 itself only run in Colab.

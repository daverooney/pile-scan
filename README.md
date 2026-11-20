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
3. **Runtime > Change runtime type**: T4 GPU (free) or L4 (Pro), with **Runtime version** on **Latest** (2026.10 or newer).
4. Fill in the settings cell, then **Runtime > Run all**. Allow Drive access when asked.

The database is written to `OUTPUT_DIR/DB_NAME` on Drive.

**llama.cpp comes prebuilt.** Each session downloads llama.cpp's own CUDA 12.8 release (about 800 MB, under a minute), so nothing is compiled in the notebook. The release needs glibc 2.38 or newer, which Colab's 2026.10 runtime (Ubuntu 24.04) has and older runtime versions don't; the notebook checks this first and tells you how to switch. It also checks that the server can see the GPU before loading the model, so a CUDA problem can't silently fall back to the CPU.

**On a T4, the first session is slower.** The release has native GPU code for newer cards (like the L4) but only portable code for the T4, so the driver compiles each kernel the first time it's used. The compiled code is saved to `CACHE_DIR/cuda-jit` on Drive after server start and after every page while it's still growing, and restored at the start of later sessions. The model files (7.4 GB for 4-bit, 12.7 GB for 8-bit) download each session unless you tick `CACHE_MODELS_ON_DRIVE`.

**When Colab disconnects**, reconnect and Run all again. The notebook copies the database back from Drive and skips every page already done. You lose at most the pages since the last checkpoint (`CHECKPOINT_EVERY`, default 10).

If this repo is private, add a Colab secret named `GITHUB_TOKEN` with a read-only token so the install cell can fetch the code.

### Settings worth knowing

| Setting | Default | Notes |
|---|---|---|
| `MODEL` | `gemma-4-12b` | `gemma-4-12b` reads pages best. `gemma-4-e4b` is about twice as fast; Google's numbers put it close on plain transcription but well behind on understanding a page. Each has its own GGUF repo from Unsloth. |
| `QUANTIZATION` | `4bit` | 12B: `4bit` is `UD-Q4_K_XL` (7.4 GB) and fits a free T4; `8bit` is `Q8_0` (12.7 GB), use an L4. E4B: 5.1 and 8.2 GB, both fit a T4. |
| `USE_MTP` | off | Multi-token prediction: Unsloth's small drafter (`MTP/mtp-…-Q8_0.gguf`) proposes tokens and the model checks them. Same text at temperature 0, usually faster. Not yet timed on a T4. |
| `MAX_SOFT_TOKENS` | `1120` | Gemma 4's visual token budget: 70, 140, 280, 560 or 1120. Passed to llama-server as `--image-min-tokens` and `--image-max-tokens`, with batch and ubatch at 2048 so an image's tokens fit in one pass. |
| `STRIPS` | `1` | Read each page's text in 2 or 3 horizontal strips. Each strip is sent with the whole page first, so the model can see column headings above it, and gets the full visual budget, so small marks are seen in more detail. Cuts go where a row of pixels crosses the least ink, so text lines aren't cut in half (`pile_scan/strips.py`). A strip inside a ruled table whose headings are higher up gets a copy of the table's heading band pasted on top, under a grey bar; a page without a ruled grid gets nothing added. A line a strip repeats from the strip above is dropped. Metadata still comes from one read of the whole page. Costs roughly one extra read per strip. Each strip's cut and reply are saved in `strips_json`, and the notebook's "Look at how a page was cut into strips" cell shows them as pictures. |
| `MAX_NEW_TOKENS` | `3072` | A page that reaches this without looping is read once more with 6144. |
| `SAMPLE_LIMIT` | `0` (all) | Stop after N pages. Files are visited in sorted order, so N always means the same pages. |
| `RETRY_ERRORS` | off | Re-read pages that failed or were cut off on an earlier run. With `SAMPLE_LIMIT` it stops after that many re-reads. |
| `LLAMA_CPP_BUILD` | `latest` | The newest llama.cpp release is downloaded once and its archives kept in `CACHE_DIR/llama.cpp/<build>-cuda<version>/`; later sessions copy it from there without checking for a newer one. Delete that folder to move to the newest release. A build such as `b12359` pins that release. |
| `CACHE_MODELS_ON_DRIVE` | off | Keep the GGUF files on Drive too. Costs Drive space, saves a download per session. |
| `DISCONNECT_WHEN_DONE` | off | Releases the GPU at the end so a finished run stops spending compute units. |

The CUDA version of the llama.cpp build is set in `pile_scan/config.py` (`LLAMA_CPP_CUDA`, `12.8`). Keep it at or below what Colab's driver supports (`nvidia-smi` shows it; 13.0 on the 2026.10 runtime). If GitHub can't be reached and nothing is cached, `LLAMA_CPP_FALLBACK_BUILD` (`b12359`) is used. A new release or driver starts a fresh T4 code cache; an L4 needs none, and the notebook says so. If a release doesn't know a server option (say `USE_MTP` on an old pinned build), the server stops with an error naming the option.

**Loops.** With greedy decoding the model can get stuck repeating itself; on ruled forms it kept writing empty table cells until it ran out of tokens. The prompt now asks for one line per filled table row (a ticked box is written as its column's heading, or `[?]` when the model can't tell which column it's in) instead of Markdown grids, with text around a table written as ordinary lines. Empty cells and rows that are only a row number are dropped from `ocr_text`; `raw_response` keeps the reply as it came. Replies are also streamed, and a reply that starts repeating is stopped at once (`pile_scan/loops.py`) and read again with llama.cpp's DRY repetition penalty. Pages still cut off after that are saved with `status = 'truncated'`.

**System RAM.** The server runs with `--cache-ram 0`. Its prompt cache otherwise keeps up to 8 GiB of past requests in RAM, which never helps when every page is different.

The first scanned page logs how many image tokens the model actually received (worked out from the server's prompt token counts). It should be close to the budget. A number near 280 means the setting didn't take effect.

### Comparing models and settings

Run the same 20 pages twice into different databases (`SAMPLE_LIMIT = 20`, for example `MODEL = gemma-4-12b` into `sample_12b.db`, then `gemma-4-e4b` into `sample_e4b.db`), then run the compare cell at the bottom of the notebook. It reports text similarity per page and whether the numbers (dates, amounts, IDs) match, since that's where smaller or more compressed models tend to slip. The same works for 4-bit against 8-bit.

## Making a test pile

`notebooks/make_test_corpus.ipynb` (CPU runtime, no GPU) makes a pile with a known answer, for scoring the scanner and, later, the sort pass. By default it downloads volume 586 of the United States Reports from govinfo.gov to Drive, takes printed pages 1-145 (the first eleven opinions, each complete), and writes:

- `CORPUS_DIR/pages/`: each page as a grayscale PNG at 200 dpi, text layer gone. `DEGRADE_SHARE` of them roughed up to look photocopied (grey paper, slight skew, speckle, sometimes an edge shadow, blur, JPEG artefacts), `ROTATE_SHARE` turned 90, 180 or 270 degrees. The order is partly shuffled: neighbours occasionally swap, about 8% of pages move in groups of 3 to 12, and about 3% land far away on their own. Files are named `scan_00001.png` and so on in pile order, so the scanner sees the shuffled order.
- `CORPUS_DIR/answer_key/`: for each image its true position, PDF page, printed page number, section, the opinion it belongs to, the rotation and roughing up applied, and its text from the PDF's text layer as ground truth. The table of cases and the index are held out here, out of the pile, for comparison with what notebook 2 rebuilds.

Point notebook 1's `INPUT_DIR` at `pages/`, never at the corpus folder. Everything random is seeded (`SEED`), and each page's treatment depends only on the seed and the page, so the same settings give the same pile. Sections are detected from running heads specific to the United States Reports (`pile_scan/testcorpus.py`); another PDF works, but its section labels won't mean much and nothing gets held out.

## What's in the database

`pages` has one row per page, keyed by `(file_sha256, page_index)`:

- `ocr_text`: the transcription. `pdf_text_layer` holds a PDF's embedded text when it has one.
- Ordering clues for the sort pass: `doc_title`, `doc_type`, `page_number_printed` (plus parsed `page_number_int` and `page_total_int` from "3 of 7"), `dates_json`, `primary_date`, `first_line`, `last_line`, `running_header`, `running_footer`, `starts_mid_sentence`, `ends_mid_sentence`, `language`.
- `answers_json`: answers to the questions in the settings cell.
- `status`: `done`, `truncated` (still cut off after a retry) or `error`. `loop_detected` and `retry_mode` (`dry` or `longer`) say whether a page had to be read twice and why. `strips_json` holds, for a page read in strips, the metadata reply and each strip's rows, heading band, reply and any lines dropped as repeats.
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

The tests cover the test-pile maker, discovery, hashing, PDF and TIFF page splitting, reply parsing, checkpoints, resume after a simulated disconnect, the runtime checks, loop detection and retries, upgrading an older database, unpacking the prebuilt release, the T4 code cache, model downloads, and the HTTP calls to llama-server (against a fake server in `tests/fake_llama_server.py`). The real GPU and Gemma 4 itself only run in Colab.

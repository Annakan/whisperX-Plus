# WhisperX Evaluator CLI - Implementation Plan

> **Status**: Draft v2 - Revised based on feedback  
> **Based on**: `Features.md`  
> **Goal**: Build a CLI tool to benchmark WhisperX transcription quality across parameter variations

---

## 1. Project Overview

### 1.1 Objective
Create an evaluator tool that:
1. Parses reference transcripts in a specific format
2. Runs WhisperX transcriptions with varying parameters (grid search)
3. Compares generated transcripts against reference using text similarity metrics
4. Aggregates results for analysis in notebooks (IPython/Marimo)

### 1.2 Key Features
- **Resumable execution**: Skip already-completed steps
- **Progress tracking**: TQDM progress bars
- **Configurable**: YAML-based run plans
- **Reproducible**: Store all parameters with outputs

---

## 2. Architecture

```
evaluator/
├── __init__.py
├── cli.py              # Main CLI entry point (Typer-based)
├── parser.py           # Reference transcript parsing
├── prepare.py          # Text normalization functions
├── runner.py           # WhisperX transcription runner
├── comparator.py       # Levenshtein/metrics calculation
├── aggregator.py       # CSV aggregation
└── models.py           # Pydantic models for config/segments
```

---

## 3. Module Specifications

### 3.1 Parser Module (`parser.py`)

**Function**: `parse_source(path: Path) -> list[Segment]`

Parses reference transcripts with two possible formats:

**Format A**: With metadata header (ignored)
```
[metadata lines - ignored]
----- (5+ dashes)
===== (5+ equals, immediately after dashes)
[optional timestamp] [<chars>] | (digits:) (digit+ ms)
+A:
Speaker A's text here
+B:
Speaker B's text here
[optional timestamp]
+C:
Speaker C's text here
...
```

**Format B**: No separator → entire file is transcript content
```
+A:
Speaker A's text here
+B:
Speaker B's text here
...
```

**Parsing Rules**:
- Look for pattern: line of 5+ `-` immediately followed by line of 5+ `=`
- If found: content after `=====` line is the transcript
- If not found: entire file is the transcript
- Timestamp lines `[<chars>] | (digits:) (digit+ ms)` are **skipped** (not parsed)

**Output**: List of `Segment(locuteur: str, text: str)`

---

### 3.2 Prepare Module (`prepare.py`)

**Function**: `prepare(segments, with_locuteur=True, with_paragraphs=True, lower_no_punct=True) -> str`

**Variations defined in Features.md**:
| Name | with_locuteur | with_paragraphs | lower_no_punct |
|------|---------------|-----------------|----------------|
| VA   | True          | True            | False          |
| VB   | False         | False           | False           |
| VC   | True          | True            | True           |
| VD   | False         | False           | True           |

---

### 3.3 Runner Module (`runner.py`)

**Function**: `run_transcription(audio_path, output_dir, params: dict) -> Path`

- Calls WhisperX transcription API/CLI
- Saves output + `params.json` in output subdirectory
- Returns path to generated transcript

**Parameter Grid** (revised):
| Parameter   | Values                    |
|-------------|---------------------------|
| model       | tiny, medium, large-v3    |
| beam_size   | 5, 7, 9                   |
| patience    | 1, 2, 3                   |
| preprocess  | 0, 1, 4                   |

**Total combinations**: 3 × 3 × 3 × 3 = **81 runs per audio file**

---

### 3.4 Comparator Module (`comparator.py`)

**Function**: `compare(generated: str, reference: str) -> dict[str, float]`

**Metrics**:
- Levenshtein distance (absolute)
- Levenshtein ratio (normalized 0-1)
- WER (Word Error Rate) - *suggested addition*
- CER (Character Error Rate) - *suggested addition*

---

### 3.5 Aggregator Module (`aggregator.py`)

**Function**: `aggregate(run_dir: Path) -> Path`

Collects all `result.csv` files and produces a consolidated CSV.

---

## 4. CLI Interface

```bash
whisperx-eval run \
    --source-dir /path/to/source/data \
    --run-dir /path/to/run/directory \
    [--stop-on-fail]      # Default: False (log errors, skip, continue)
    [--sample N]          # Run only N random parameter combinations
    [--dry-run]           # Validate and show plan without executing
```

### 4.1 runplan.yaml Schema

```yaml
to_process:
  - subdir1
  - subdir2

base_args:
  language: fr
  device: cuda
  compute_type: float16
  diarize: true

varying_args:
  model: [tiny, medium, large-v3]
  beam_size: [5, 7, 9]
  patience: [1, 2, 3]
  preprocess: [0, 1, 4]
```

---

## 5. Output Structure

```
run_dir/
├── runplan.yaml
├── subdir1/
│   ├── audio_*.ext                  # Copied source (pattern: audio_*.*)
│   ├── *_REF.txt                    # Copied reference (pattern: *_REF.txt)
│   ├── model=small_beam=5_patience=1_preprocess=0/
│   │   ├── params.json
│   │   ├── transcript.json
│   │   ├── VA.txt, VB.txt, VC.txt, VD.txt
│   │   └── result.csv
│   └── ... (other variations)
├── subdir2/
│   └── ...
└── aggregated_results.csv           # Final consolidated CSV
```

---

## 6. Suggested Libraries

| Purpose | Library | Rationale |
|---------|---------|-----------|
| CLI framework | **Typer** | Consistent with existing whisperX CLI |
| Config parsing | **Pydantic + PyYAML** | Type-safe YAML handling |
| Progress bars | **tqdm** | As specified |
| Text comparison | **rapidfuzz** | Faster than difflib, C-based |
| WER/CER metrics | **jiwer** | Standard ASR evaluation library |
| Diff visualization | **diff-match-patch** | As specified, for detailed diffs |
| Data handling | **pandas** | CSV operations, aggregation |

---

## 7. Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load runplan.yaml                                        │
├─────────────────────────────────────────────────────────────┤
│ 2. For each source in to_process:                           │
│    ├─ Find audio (audio_*.*) and reference (*_REF.txt)      │
│    ├─ Copy audio + reference to run_dir                     │
│    ├─ Parse reference transcript                            │
│    └─ For each parameter combination (or --sample N):       │
│        ├─ [SKIP if output exists]                           │
│        ├─ Run transcription                                 │
│        │   └─ On error: log + skip (or abort if --stop-on-fail) │
│        ├─ Parse generated transcript                        │
│        ├─ Normalize speaker labels (first→A, second→B, etc) │
│        ├─ Generate VA/VB/VC/VD variants                     │
│        ├─ Compare each variant                              │
│        └─ Write result.csv                                  │
├─────────────────────────────────────────────────────────────┤
│ 3. Aggregate all result.csv → aggregated_results.csv        │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Resolved Questions

| # | Question | Resolution |
|---|----------|------------|
| Q1 | Reference format parsing | Line of 5+ `-` immediately followed by 5+ `=` → content after is transcript. If no such pattern, entire file is transcript. |
| Q2 | Timestamp lines | Skip entirely (not parsed) |
| Q3 | Speaker mapping | First speaker occurrence in WhisperX output → A, second → B, etc. Do a normalization pass before comparison. |
| Q4 | VB variation | Corrected to: `with_locuteur=False, with_paragraphs=False, lower_no_punct=False` |
| Q5 | Grid size | Reduced to 81 combinations. Sample mode supported via `--sample N`. No parallel for now. |
| Q6 | Speaker mapping strategy | Order-of-appearance mapping (not label matching) |
| Q7 | File discovery | Audio: `audio_*.*` (only one expected). Reference: `*_REF.txt` (only one expected). |
| Q8 | Error handling | `--stop-on-fail` option (default False). If False: log error, skip, continue. If True: abort. |

---

## 9. Proposed Enhancements

### 9.1 Additional Metrics
Beyond Levenshtein, ASR evaluation commonly uses:
- **WER (Word Error Rate)**: Industry standard, accounts for insertions/deletions/substitutions
- **CER (Character Error Rate)**: More granular
- **MER (Match Error Rate)**: Alternative to WER

**Recommendation**: Add WER/CER using the `jiwer` library.

### 9.2 Structured Output
Store comparison results as JSON in addition to CSV for richer data:
```json
{
  "params": {...},
  "metrics": {
    "VA": {"levenshtein": 0.95, "wer": 0.12, "cer": 0.05},
    ...
  },
  "runtime_seconds": 45.2
}
```

### 9.3 Dry-Run Mode
```bash
whisperx-eval run --dry-run ...
```
- Show what would be executed
- Calculate total estimated time
- Validate runplan.yaml

### 9.4 Sample Mode ✓ (Confirmed)
```bash
whisperx-eval run --sample 3 ...
```
- Run only N random parameter combinations for quick validation
- **Status**: Will be implemented

### 9.5 HTML Diff Report
Generate visual diff reports using `diff-match-patch` for qualitative analysis.

### 9.6 Logging Configuration
Separate log levels for:
- Progress output (always visible)
- Transcription details (verbose)
- Debug information

---

## 10. Dependencies (proposed `pyproject.toml` additions)

```toml
[project.optional-dependencies]
evaluator = [
    "typer>=0.9.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "tqdm>=4.66",
    "rapidfuzz>=3.0",
    "jiwer>=3.0",
    "diff-match-patch>=20230430",
    "pandas>=2.0",
]
```

---

## 11. Implementation Phases

### Phase 1: Core Infrastructure (MVP)
- [ ] Parser module with tests
- [ ] Prepare module with tests
- [ ] Pydantic models for configuration
- [ ] Basic CLI skeleton

### Phase 2: Transcription Runner
- [ ] Integration with whisperX transcription API
- [ ] Parameter grid generation
- [ ] Skip-if-exists logic
- [ ] params.json output

### Phase 3: Comparison & Aggregation
- [ ] Comparator module (Levenshtein + WER/CER)
- [ ] Per-variation result.csv generation
- [ ] Aggregation to final CSV

### Phase 4: Polish
- [ ] Progress bars (tqdm)
- [ ] Error handling with `--stop-on-fail` option
- [ ] Logging configuration
- [ ] Dry-run mode
- [ ] Sample mode (`--sample N`)
- [ ] Documentation

### Phase 5: Enhancements (Optional)
- [ ] HTML diff reports
- [ ] Parallel execution support

---

## 12. Open Decisions

| Decision | Options | Resolution |
|----------|---------|------------|
| Invoke whisperX as | subprocess CLI / Python API | **Python API** (cleaner) |
| Store diffs | Yes / No | TBD (optional enhancement) |
| Speaker mapping | Ignore / Best-match / Manual | **Order-of-appearance** (first→A, second→B) |
| Parallelization | None / multiprocessing / async | **None** (out of scope for now) |
| Error handling | Abort / Skip | **Configurable** via `--stop-on-fail` |

---

---

## 13. Speaker Normalization Algorithm

To map WhisperX speaker labels to sequential letters:

```python
def normalize_speakers(segments: list[Segment]) -> list[Segment]:
    """
    Replace SPEAKER_XX labels with A, B, C... based on order of first appearance.
    """
    speaker_map: dict[str, str] = {}
    next_letter = ord('A')
    
    for segment in segments:
        if segment.speaker not in speaker_map:
            speaker_map[segment.speaker] = chr(next_letter)
            next_letter += 1
        segment.locuteur = speaker_map[segment.speaker]
    
    return segments
```

**Example**:
- Input: `SPEAKER_02, SPEAKER_00, SPEAKER_02, SPEAKER_01`
- Output: `A, B, A, C`

---

*Document revised based on feedback. Pending final review before implementation.*

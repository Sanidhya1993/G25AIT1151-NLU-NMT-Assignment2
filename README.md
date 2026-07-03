# Sanskrit → English Neural Machine Translation

**Course:** Natural Language Understanding — Assignment 2
**Roll No:** G25AIT1151

A sequence-to-sequence NMT system that translates Sanskrit into English. The model is a Transformer encoder–decoder initialised from the disclosed pretrained checkpoint [`ai4bharat/indictrans2-indic-en-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M) and adapted to the task with **LoRA** (Low-Rank Adaptation) — only low-rank adapter matrices are trained (~5.8% of the parameters), then merged back into the base weights for inference. Fine-tuning uses the provided parallel data only. No external translation API is used.

## Results (public split)

| Split | BLEU (NLTK) | BERTScore F1 | Inference time | Parameters |
|-------|-------------|--------------|----------------|------------|
| Dev (1000)  | 0.2611 | 0.6213 | 128.7 s | 211,780,608 |
| Test (1000) | 0.2500 | 0.6216 | 140.8 s | 211,780,608 |

BLEU is the default NLTK `corpus_bleu` (uniform 4-gram weights, no smoothing). BERTScore is F1 with `rescale_with_baseline=True`. Inference time is total wall-clock for the full 1000-sentence split on an RTX 3060 Laptop GPU with KV caching enabled. The merged model has 211,780,608 parameters in total; only **12,976,128 (5.77%)** were trained via LoRA — the rest of the base checkpoint stays frozen.

## Repository contents

| File | Description |
|------|-------------|
| `G25AIT1151_NLU_NMT_v8.ipynb` | Full pipeline: data loading, preprocessing, LoRA fine-tuning, evaluation, submission generation |
| `submission.csv` | Test-set predictions — columns `Source_id`, `Sentence_en` (UTF-8) |
| `metrics.json` | BLEU, BERTScore F1, inference time, parameter count (dev + test) |
| `G25AIT1151_NLU_NMT_Report.pdf` | Report |
| `README.md` | This file |

## Requirements

- **Python 3.10 or 3.11** (avoid 3.13).
- A CUDA-capable GPU is recommended. Training and the reported timings used an **NVIDIA RTX 3060 Laptop GPU, CUDA 12.6**; the notebook falls back to CPU if no GPU is present (much slower).
- The provided dataset CSVs placed where the notebook can find them (see [Dataset](#dataset)).

## Environment setup

The commands below create an isolated virtual environment, install dependencies, register a Jupyter kernel, and launch the notebook. They are written for **Windows (PowerShell / CMD)**; equivalent notes for macOS/Linux follow.

### Windows

```bat
:: 1. Create and activate a Python 3.11 virtual environment
py -3.11 -m venv .venv
.venv\Scripts\activate
python --version

:: 2. Upgrade pip
python -m pip install --upgrade pip

:: 3. Jupyter tooling
pip install notebook ipykernel

:: 4. PyTorch with CUDA 12.6 (use the CPU build if you have no GPU — see note below)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

:: 5. Core dependencies (versions pinned for reproducibility)
pip install transformers==4.40.2 tokenizers==0.19.1 accelerate==0.30.1 peft==0.11.1 sentencepiece sacremoses bert-score nltk pandas ipywidgets

:: 6. IndicTrans2 preprocessing toolkit
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

:: 7. Register the Jupyter kernel
python -m ipykernel install --user --name=nlu-final --display-name="NLU Final Python 3.11"

:: 8. Launch Jupyter
jupyter notebook
```

## Dataset

Parallel Sanskrit–English CSVs, aligned on `Source_id`:

```
train_sa.csv  train_en.csv     # 9,982 aligned pairs after cleaning
dev_sa.csv    dev_en.csv       # 1,000
test_sa.csv   (test_en.csv)    # 1,000  (test_en optional — see below)
```

The loader searches the working directory, `/mnt/data`, and common Colab Drive paths, and accepts both the plain names (`train_sa.csv`) and the numbered variants (`train_sa_10000.csv`). Place the CSVs next to the notebook or adjust the search roots in the data-loading cell.

**Private-evaluation mode.** `test_en.csv` (gold references) is optional. If it is absent, the notebook still runs end-to-end, generates `submission.csv`, and simply skips the test-set BLEU/BERTScore computation — so the same notebook works when only the private test source is released.

## How to run

1. Complete the environment setup and select the **NLU Final Python 3.11** kernel.
2. Ensure the dataset CSVs are reachable.
3. **Run all cells top to bottom.** The notebook will:
   - load and align the data;
   - load the base model, inject LoRA adapters, and fine-tune (dev-BLEU checkpoint selection);
   - merge the adapters into the base weights and save the best model to `indictrans2_clean_best_model/`;
   - reload the best checkpoint and evaluate on dev;
   - translate the test set and write `submission.csv`;
   - compute metrics and write `metrics.json`.

To **evaluate only** (skip training), run the setup cells, then the "Load best fine-tuned checkpoint" cell, then the evaluation/generation cells — this reuses the saved (merged) checkpoint in `indictrans2_clean_best_model/`.

> **Switching to full fine-tuning.** LoRA is enabled by default (`USE_LORA = True` in the config cell). Set `USE_LORA = False` to run a full fine-tune of all 211M parameters instead; the rest of the pipeline is unchanged. In these experiments LoRA gave higher BLEU and BERTScore on both splits (see the report's Experiments section).

## Pre-trained models disclosed

- `ai4bharat/indictrans2-indic-en-dist-200M` — translation model (fine-tuned on provided data only, via LoRA adapters using the Hugging Face `peft` library).
- The `bert-score` library uses a pretrained contextual encoder for **evaluation only** (baseline rescaling); it is not part of the translation model.

No external datasets or translation APIs are used.

## Reproducibility notes

- All randomness is seeded (`SEED = 42`); minor run-to-run variance (~±0.001 BLEU) can remain due to GPU non-determinism.
- Label-smoothed training loss reads higher in absolute terms than plain cross-entropy — judge runs by **dev BLEU**, not loss magnitude.
- LoRA training reports the adapter total (~224.76M parameters, 12.98M trainable) during fine-tuning; after merging, the deployed model is a standard 211,780,608-parameter Transformer.
- Pinned versions: `transformers==4.40.2`, `tokenizers==0.19.1`, `accelerate==0.30.1`, `peft==0.11.1`.

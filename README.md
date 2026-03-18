# Medcompreviser

Medcompreviser is a local LLM pipeline for rewriting patient education materials (PEMs) into plain language (≈6th-grade reading level), with verification and glossary generation.

This repo currently supports:

- Readability-focused rewriting using a local LLM (vLLM + Qwen)
- Sentence-level verification (hallucination + dropped content checks)
- Glossary refinement for difficult medical terms

---

## Project Structure

Medcompreviser/
    scripts/
        start_qwen.sh # Start local LLM server
        run_pipeline.py # Run rewrite pipeline
    src/medcompreviser/
        llm.py
        rewrite.py
        verify.py
        definitions.py
        eval.py
    data/de_novo/
    outputs/

---

## Setup

### 1. Create environment

```bash
python -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Request a GPU session (HPC)

srun --account=accountname --partition=spgpu --gpu=1 --mem=24G --cpus-per-task=2 --time=02:00:00 --pty bash


### 3. Start the model server

```
bash scripts/start_qwen.sh
```

Wait unitil you see: Application startup complete


### 4. Run the pipeline

In another terminal on the same node:

```
source .venv312/bin/activate
python scripts/run_pipeline.py
```

**Output**
The pipeline prints:
- number of attempts
- source readability
- final readability
- rewritten text
- glossary
- verification summary

Future versions will save structured JSON outputs to /outputs.



## What the Pipeline Does

- Rewrite
- Converts PEM text to ~6th grade reading level
- Verify
- Maps rewritten sentences to source sentences
- Flags:
    - unsupported content
    - dropped instructions
    - numeric mismatches (e.g., time/dosage errors)
- Definitions
    - Identifies difficult medical terms
    - Generates plain-language definitions


### Notes 
- First model run may take time (downloads weights)
- Make sure cache directories are set to scratch (see start_qwen.sh)
- This version focuses on readability only (no personalization yet)

### Planned Next Steps
- Personalization module (patient-specific adaptation)
- Improved verification (semantic entailment)
- Visual augmentation
- Batch processing + structured outputs


### Troubleshooting

Module not found (medcompreviser)
        pip install -e .

Port 8000 connection refused
        Model server not running or crashed
        Check logs or restart start_qwen.sh

No space left on device
        Ensure HuggingFace cache is redirected to scratch



### License
MIT License
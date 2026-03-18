# Medcompreviser

Medcompreviser is a locally hosted vLLM server (Qwen2.5-14B) pipeline for rewriting patient education materials (PEMs) into plain language (≈6th-grade reading level), with verification and glossary generation. It reads a PDF
and writes a JSON artifact to outputs.

The pipeline includes an optional semantic verification step using an NLI model.
By default it uses facebook/bart-large-mnli, which Hugging Face documents as a checkpoint trained on MultiNLI and explicitly demonstrates for entailment-style classification by comparing a premise and a hypothesis.
This step checks whether rewritten sentences are semantically supported by their mapped source sentences.

This repo currently supports:

- Readability-focused rewriting using a local LLM (vLLM + Qwen)
- Sentence-level verification (hallucination + dropped content checks)
- Glossary refinement for difficult medical terms

---

## Project Structure
```
```Medcompreviser/
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
    outputs/ ```

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


### 3. Start the model server (choose a port, e.g., 8100)

```
bash scripts/start_qwen.sh 8100
```

Wait unitil you see: Application startup complete


### 4. Run the pipeline

In another terminal, ssh into the same node, run:

```
source .venv312/bin/activate
export VLLM_BASE_URL=http://127.0.0.1:8100/v1
```

Check that it is set correctly:
```
bash scripts/check_server.sh
```

Then run;
```
python scripts/run_pipeline.py \
  --input data/de_novo/1.pdf \
  --output outputs/1_sample_run.json \
  --target-grade 6 \
  --model-name qwen14b
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

*Module not found (medcompreviser)*
        
        pip install -e .

*Port 8000 connection refused*
        
        Model server not running or crashed
        Check logs or restart start_qwen.sh

*No space left on device*
        
        Ensure HuggingFace cache is redirected to scratch
    
*Port Conflict* 

If you see an error like 
           ``` OSError: [Errno 98] Address already in use ```
it means the port is already being used (often by another user on the same node). Solution is to use another port.    
    Eg.     ```bash scripts/start_qwen.sh 8101
            export VLLM_BASE_URL=http://127.0.0.1:8101/v1 ```

Note that 
- The port must match between:
    - start_qwen.sh
    - VLLM_BASE_URL
- Default is 8000, but using 8100+ is recommended on shared systems.
- The server must be running before executing run_pipeline.py.

### License
MIT License

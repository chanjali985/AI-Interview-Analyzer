# Legacy CLI (v1)

This folder holds the original project exactly as it was before the platform rewrite: a
command-line tool and a single-endpoint FastAPI service that analysed **one** audio answer at a
time against a resume file.

It is kept for reference and for one-off local analyses. It is **not** wired into the platform,
has no database, no email and no frontend, and it is not covered by the test suite.

To run it, work inside this folder so its imports resolve:

```bash
cd legacy
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

ollama serve &          # in another terminal
ollama pull llama3.2

python main.py \
  --question "Tell me about your Python experience" \
  --audio ../examples/Audio.m4a \
  --resume ../examples/sample_resume.txt \
  --output results.json
```

The original documentation is in [`README_original.md`](README_original.md).

For anything real — multiple questions, candidate invitations, stored results, emailed
reports — use the platform in the repository root instead.

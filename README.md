# AI Interview Analyzer

A production-ready, self-hosted **asynchronous interview platform**. Recruiters define a role and
its question set, invite candidates by email, and candidates answer by voice in their browser.
Every answer is transcribed, scored on five dimensions, matched against the candidate's resume,
and turned into a report that is **emailed to the candidate and the hiring team automatically**.

Runs entirely on free local models by default (Whisper + Ollama), or against a hosted API if you
prefer speed over cost.

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Quick start with Docker](#quick-start-with-docker)
- [Quick start without Docker](#quick-start-without-docker)
- [Setting up email (Gmail)](#setting-up-email-gmail)
- [Using the platform](#using-the-platform)
- [How scoring works](#how-scoring-works)
- [Configuration reference](#configuration-reference)
- [API](#api)
- [Testing and CI](#testing-and-ci)
- [Deploying to production](#deploying-to-production)
- [Security and privacy](#security-and-privacy)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)

---

## What it does

**For the hiring team**

- Create job roles, each with its own ordered question set and per-question time limits.
- Invite a candidate by email — they get a private, expiring link. No account, no scheduling.
- Watch the pipeline on a dashboard: invited → in progress → submitted → analysing → completed.
- Open any candidate's report: overall score, five sub-scores, resume/answer skill match,
  strengths, areas to improve, full transcripts, and the original audio for every answer.
- Get every finished report by email, plus an alert when an analysis fails.
- Re-run an analysis or re-send a report without asking the candidate to record again.

**For the candidate**

- Open the link, read what to expect, upload or paste a resume.
- Answer each question by voice with a live level meter and a countdown; listen back and
  re-record before saving.
- Submit once, then watch the status; the feedback report arrives by email with a PDF attached.

---

## Architecture

```
                    ┌──────────────────────────────┐
  Recruiter ───────▶│  React SPA (Vite + Tailwind) │◀─────── Candidate
                    │  dashboard · roles · reports │      (invite-token link,
                    │  candidate recording flow    │       browser recording)
                    └───────────────┬──────────────┘
                                    │ JSON / multipart over HTTPS
                    ┌───────────────▼──────────────┐
                    │  FastAPI                     │
                    │  JWT auth · signed invites   │
                    │  validation · rate limiting  │
                    └───────────────┬──────────────┘
                                    │ enqueue
                    ┌───────────────▼──────────────┐
                    │  Analysis worker (thread pool)│
                    │  1 transcribe  faster-whisper │
                    │  2 score       LLM (5 dims)   │
                    │  3 skills      LLM extraction │
                    │  4 relevance   Sentence-BERT  │
                    │  5 summary     LLM narrative  │
                    └───────┬───────────────┬───────┘
                            │               │
                  ┌─────────▼──────┐  ┌─────▼───────────────┐
                  │ SQLite/Postgres│  │ SMTP (Gmail)        │
                  │ + audio files  │  │ candidate + recruiter│
                  └────────────────┘  │ HTML mail + PDF      │
                                      └──────────────────────┘
```

Everything behind an interface: the LLM, the speech-to-text engine and the embedding model are
each pluggable, and each degrades safely rather than taking an interview down.

| Layer | Default (free, local) | Alternatives |
|---|---|---|
| LLM | Ollama + `llama3.2` | OpenAI, Groq, Gemini (`LLM_PROVIDER`) |
| Speech-to-text | faster-whisper `base`, CPU | OpenAI Whisper API, Google free endpoint |
| Skill relevance | Sentence-BERT `all-MiniLM-L6-v2` | lexical fallback if the model is absent |
| Database | SQLite | PostgreSQL (`DATABASE_URL`) |
| Email | Gmail SMTP | any SMTP server |

---

## Quick start with Docker

The fastest path to a working system.

```bash
git clone https://github.com/chanjali985/AI-Interview-Analyzer.git
cd AI-Interview-Analyzer

cp .env.example .env
# Edit .env — at minimum set:
#   SECRET_KEY        (python -c "import secrets; print(secrets.token_urlsafe(48))")
#   ADMIN_EMAIL / ADMIN_PASSWORD
#   SMTP_PASSWORD     (Gmail App Password — see the email section below)
#   PUBLIC_BASE_URL   (the URL candidates will open)

docker compose up -d --build

# Pull the language model once (~2 GB, cached in a volume afterwards)
docker compose run --rm model-init
```

Open <http://localhost:8000> and sign in with the `ADMIN_EMAIL` / `ADMIN_PASSWORD` from your `.env`.

Check everything is wired up:

```bash
curl http://localhost:8000/api/health
# {"status":"ok", ..., "llm_ready":true, "transcription_ready":true, "mail_configured":true}
```

> The image builds the frontend, installs ffmpeg and pre-downloads the Whisper and embedding
> models so the first interview is not slow. If your build environment cannot reach
> huggingface.co, build with `--build-arg SKIP_MODEL_WARMUP=1` and the models download on
> first use instead.

---

## Quick start without Docker

**Prerequisites:** Python 3.11+, Node 20+, ffmpeg, and [Ollama](https://ollama.com).

```bash
# 1. Language model
ollama serve &            # keep running
ollama pull llama3.2

# 2. Everything else
make setup                # venv + backend deps + frontend deps + .env
# edit .env (SECRET_KEY, ADMIN_PASSWORD, SMTP_PASSWORD, PUBLIC_BASE_URL)

# 3. Run it
make backend              # API on http://localhost:8000
make frontend             # dev UI on http://localhost:5173 (proxies /api to :8000)
```

For a single-process production-style run, build the frontend once and let FastAPI serve it:

```bash
make build
make backend              # http://localhost:8000 now serves the UI too
```

---

## Setting up email (Gmail)

This is the part that emails the score to the candidate, so it is worth getting right.
**Gmail will not accept your normal password** — you need an App Password.

1. Enable 2-Step Verification: <https://myaccount.google.com/security>
2. Create an App Password: <https://myaccount.google.com/apppasswords>
   (choose "Mail" → "Other", name it "Interview Analyzer")
3. Copy the 16 characters and **remove the spaces**.
4. Put it in `.env`:

```env
MAIL_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_STARTTLS=true
SMTP_USERNAME=your-address@gmail.com
SMTP_PASSWORD=abcdefghijklmnop          # the 16-char App Password
MAIL_FROM=your-address@gmail.com
MAIL_FROM_NAME="Your Company Hiring"
RECRUITER_NOTIFY_EMAIL=your-address@gmail.com
```

Verify it without sending anything to a candidate:

```bash
# signed in as a recruiter, with $TOKEN from /api/auth/login
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/health/email
curl -X POST -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/health/email/test
```

**Three emails are sent automatically:**

| When | To | Contains |
|---|---|---|
| A candidate is invited | candidate | Role summary, what to expect, private interview link |
| An analysis completes | candidate | Overall score, five sub-scores, skill match, strengths, growth areas, PDF report |
| An analysis completes | `RECRUITER_NOTIFY_EMAIL` | Same scores plus a link to the full report and transcripts |
| An analysis fails | `RECRUITER_NOTIFY_EMAIL` | The error and a link to re-run — the candidate is *not* emailed |

Set `SEND_SCORES_TO_CANDIDATE=false` if you want scores to stay internal; candidates still get
the invitation email.

Gmail's free tier allows roughly 500 messages a day. For higher volume point `SMTP_*` at
SendGrid, Amazon SES, Postmark or your own relay — nothing else changes.

---

## Using the platform

**1. Create a role.** *Roles & questions → New role.* Give it a title, an optional description
(candidates see it), and your questions with time limits. A starter role is created on first run
so you can try the flow immediately.

> Once a candidate has been invited to a role, its questions lock. Two candidates for the same
> role always answer the same questions, which is what makes their scores comparable. Need a
> different set? Create a new role.

**2. Invite a candidate.** *Candidates → Invite a candidate.* Name, email, role. If you already
have their resume you can paste it; otherwise they upload one themselves. The email goes out
immediately, and the link is shown so you can share it another way if you prefer.

**3. The candidate records.** They open the link, allow the microphone, answer each question,
listen back, re-record if they want, and submit. Nothing is analysed until they submit.

**4. Analysis runs in the background.** Roughly 20–60 seconds per answer on CPU. The dashboard
and the candidate's status page both poll, so no refreshing.

**5. Review.** Open the candidate to see scores, transcripts, per-answer feedback and the audio.
Both parties already have the report by email.

---

## How scoring works

Each answer is graded 1–10 on five dimensions by the language model:

| Dimension | What it measures |
|---|---|
| Communication | Structure and articulation |
| Technical relevance | Whether the answer actually addresses the question, correctly |
| Confidence | Conviction and ownership |
| Clarity | How easy the answer is to follow |
| Overall quality | Holistic judgement |

Dimensions are averaged across answers, then folded into one score with configurable weights
(default: technical relevance 30%, communication 20%, overall 20%, confidence 15%, clarity 15%).

Separately, skills are extracted from the resume and from the transcripts, embedded with
Sentence-BERT, and compared by cosine similarity to give a **relevance score** (0–1). That
nudges the final score by `RELEVANCE_WEIGHT` (default 15%) — talking about what is on your
resume counts for something, but it cannot rescue a weak answer.

The final score maps to a verdict against `SHORTLIST_THRESHOLD` (default 7.0):

| Score | Verdict |
|---|---|
| ≥ 8.5 | Strong match |
| ≥ 7.0 | Shortlisted |
| ≥ 5.0 | Needs review |
| < 5.0 | Not a match right now |

**These are decision support, not decisions.** Scores come from a language model reading an
imperfect transcript. Every email and PDF says so, and a person should review the transcript
before acting. Tune the weights to your own bar, and re-run analyses after changing them.

---

## Configuration reference

Every setting lives in `.env` — see [`.env.example`](.env.example) for the annotated list. The
ones that matter most:

| Variable | Default | Notes |
|---|---|---|
| `PUBLIC_BASE_URL` | `http://localhost:8000` | Invite links are built from this — must be reachable |
| `SECRET_KEY` | random per boot | **Set it.** Otherwise every restart invalidates sessions and invite links |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | `admin@example.com` / `change-me-now` | Bootstrap recruiter, created on first start |
| `DATABASE_URL` | `sqlite:///./data/interviews.db` | Use Postgres for anything shared |
| `LLM_PROVIDER` / `LLM_MODEL` | `ollama` / `llama3.2` | `openai`, `groq`, `gemini` also supported |
| `TRANSCRIPTION_PROVIDER` | `faster_whisper` | `openai` for the hosted Whisper API |
| `WHISPER_MODEL` | `base` | `tiny` is ~3× faster, `small`/`medium` are more accurate |
| `SMTP_PASSWORD` | — | Gmail App Password. Without it, no emails are sent |
| `RECRUITER_NOTIFY_EMAIL` | — | Gets every report and every failure alert |
| `SEND_SCORES_TO_CANDIDATE` | `true` | `false` keeps scores internal |
| `ANALYSIS_WORKERS` | `2` | Interviews analysed concurrently |
| `SHORTLIST_THRESHOLD` | `7.0` | Where "shortlist" starts |
| `RATE_LIMIT_SUBMISSIONS_PER_HOUR` | `20` | Per client IP, on candidate uploads |

---

## API

Interactive docs at `/docs` in non-production environments. All paths are prefixed with `/api`.

**Recruiter — `Authorization: Bearer <token>`**

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/auth/login` | Exchange email + password for a JWT |
| `GET` | `/auth/me` | Current user |
| `POST` | `/auth/users` | Create another recruiter (superuser only) |
| `GET/POST` | `/roles` | List / create roles with question sets |
| `GET/PATCH/DELETE` | `/roles/{id}` | Read, update, archive |
| `POST` | `/interviews/invite` | Create an interview and email the link |
| `GET` | `/interviews` | List with `status`, `role_id`, `search` filters |
| `GET` | `/interviews/{id}` | Full report with transcripts |
| `GET` | `/interviews/{id}/report.pdf` | Download the PDF |
| `GET` | `/interviews/{id}/answers/{answer_id}/audio` | Stream a recording |
| `POST` | `/interviews/{id}/reanalyze` | Re-run analysis on stored recordings |
| `POST` | `/interviews/{id}/resend-report` | Email the report again |
| `GET` | `/dashboard` | Aggregate stats |

**Candidate — `X-Interview-Token: <invite token>`**

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/public/session` | Role, questions, candidate details |
| `POST` | `/public/session/start` | Mark the interview started |
| `POST` | `/public/resume` | Upload a resume file or paste text |
| `POST` | `/public/answers/{question_id}` | Upload one recorded answer |
| `POST` | `/public/submit` | Finalise and queue the analysis |
| `GET` | `/public/status` | Poll analysis progress |

**Health** — `GET /api/health`, `/api/health/live`, `/api/health/ready`, and (authenticated)
`/api/health/email`.

---

## Testing and CI

```bash
make test           # 57 tests: auth, roles, the whole candidate flow, analysis,
                    # scoring maths, email rendering, PDF generation, tokens,
                    # rate limiting and upload validation
make lint           # ruff
cd frontend && npm run build
```

Tests stub the LLM, Whisper and SMTP, so the suite runs in seconds with no models, no network
and no mail server. GitHub Actions runs backend lint + tests, the frontend build, and a Docker
image build on every push and pull request.

---

## Deploying to production

1. **Set `ENVIRONMENT=production`.** This switches logs to JSON and hides `/docs`.
2. **Set a real `SECRET_KEY`** and a strong `ADMIN_PASSWORD`. The app logs an error at startup
   if the default password is still in place.
3. **Terminate TLS in front of the app** (nginx, Caddy, a cloud load balancer) and set
   `PUBLIC_BASE_URL` to the `https://` address. Microphone access requires a secure context —
   browsers block recording on plain HTTP except on `localhost`.
4. **Use Postgres** and back it up. Audio and resumes live under `STORAGE_DIR`; back that up too
   or point it at durable storage.
5. **Keep one web worker** (`WEB_CONCURRENCY=1`) unless you move the queue out of process.
   Analysis runs in a thread pool inside the web process; raise `ANALYSIS_WORKERS` to scale it.
   Before scaling to multiple replicas, swap `app/worker.py:enqueue` for Celery or RQ — the
   interface is one function.
6. **Size the box for Whisper.** `base` on CPU needs ~2 GB RAM and runs roughly at real time.
   Give it 4 GB and 2 cores per analysis worker, or point `TRANSCRIPTION_PROVIDER=openai` at
   the hosted API if you would rather pay than provision.
7. **Watch `/api/health/ready`** — it returns 503 when the database, model or transcription
   backend is not reachable, which is what your orchestrator should gate traffic on.

---

## Security and privacy

- Recruiter sessions are JWTs; passwords are bcrypt-hashed. Login responses never reveal whether
  an account exists.
- Candidate links are signed, typed and expiring JWTs — a recruiter token cannot be used as an
  invite and vice versa. Treat the link as a secret: anyone holding it can take that interview.
- Uploads are checked for extension and size and streamed to disk; filenames are sanitised and
  path traversal is blocked by tests.
- Candidate-supplied values are HTML-escaped in every email template.
- Candidate endpoints are rate-limited per IP. The limiter is per-process — put a shared limiter
  or WAF in front of a multi-replica deployment.
- Deleting an interview deletes its recordings from disk.
- You are storing voice recordings, resumes and automated assessments of real people. Tell
  candidates what you collect and how long you keep it, honour deletion requests, and check
  your local rules on automated decision-making in hiring (GDPR Art. 22, NYC Local Law 144,
  the EU AI Act and similar) before using scores to reject anyone.

---

## Troubleshooting

**`llm_ready: false` in `/api/health`** — Ollama is not running or the model is not pulled.
`curl http://localhost:11434/api/tags`, then `ollama pull llama3.2`. In Docker,
`docker compose run --rm model-init`.

**`transcription_ready: false`** — `faster-whisper` is not installed (`pip install -r
backend/requirements-ml.txt`) or the model could not download. Check the container has network
access to huggingface.co on first run.

**Emails are not arriving** — `GET /api/health/email` tells you exactly what SMTP said. Almost
always a missing or wrong App Password. Also check the candidate's spam folder, and note that
Gmail throttles bulk sending from consumer accounts.

**"Microphone access was blocked"** — the page must be served over HTTPS (or `localhost`), and
the candidate must allow the permission prompt. Safari also requires a user gesture, which the
Start button provides.

**Analysis failed with "None of the recordings could be transcribed"** — usually genuinely silent
recordings (wrong input device), or ffmpeg missing. `ffmpeg -version` inside the container.
Fix the cause, then use **Re-run analysis** — the audio is still there.

**Scores look low across the board** — `llama3.2` grades conservatively. Try a larger model,
adjust the weights, or lower `SHORTLIST_THRESHOLD` to match your bar.

---

## Project layout

```
.
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app factory, middleware, SPA serving
│   │   ├── config.py          all settings, loaded from .env
│   │   ├── database.py        engine, sessions, init
│   │   ├── models.py          User, Role, Question, Candidate, Interview, Answer
│   │   ├── schemas.py         request/response contracts
│   │   ├── security.py        bcrypt + JWT (access and invite tokens)
│   │   ├── deps.py            auth dependencies
│   │   ├── ratelimit.py       per-IP sliding window
│   │   ├── worker.py          background analysis queue
│   │   ├── seed.py            first-run admin and starter role
│   │   ├── routers/           auth, roles, interviews, public, health
│   │   ├── services/
│   │   │   ├── llm.py         pluggable LLM providers + JSON extraction
│   │   │   ├── transcription.py  faster-whisper / Whisper API / Google
│   │   │   ├── embeddings.py  Sentence-BERT with lexical fallback
│   │   │   ├── analyzer.py    the five-step analysis pipeline
│   │   │   ├── pipeline.py    analyse → persist → notify
│   │   │   ├── resume.py      PDF/DOCX/TXT extraction
│   │   │   ├── storage.py     validated uploads
│   │   │   ├── mailer.py      SMTP + Jinja templates
│   │   │   ├── notifications.py  the four transactional emails
│   │   │   └── report_pdf.py  the PDF report
│   │   └── templates/emails/  invite, candidate report, recruiter report, failure
│   └── tests/                 pytest suite
├── frontend/
│   └── src/
│       ├── pages/             Login, Dashboard, Roles, RoleEditor, Interviews,
│       │                      Report, Interview (candidate), NotFound
│       ├── components/        Layout and the shared UI kit
│       └── lib/               API client, auth context, recorder hook, formatting
├── legacy/                    the original CLI and single-shot API, kept for reference
├── Other_questions_code/      unrelated exercise scripts from the original repo
├── Dockerfile                 multi-stage build (frontend + backend + models)
├── docker-compose.yml         app + ollama + postgres
└── .env.example               annotated configuration
```

The original CLI still works if you want a one-off analysis without the platform — see
[`legacy/README_original.md`](legacy/README_original.md).

---

## Licence

Add the licence you intend to ship under. Note that the default model weights
(Whisper, Llama 3.2, all-MiniLM-L6-v2) carry their own licences.

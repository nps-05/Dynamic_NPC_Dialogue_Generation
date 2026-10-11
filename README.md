# Nick Valentine Dialogue Generator

Local web app that generates short, in-character replies as Nick Valentine. A Flask server runs [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on your machine and can retrieve similar lines from a ChromaDB index you build yourself.

There is no hosted API and no per-request cost. The first run downloads the model (about 3 GB) and needs roughly 6 GB of RAM. On CPU, a reply usually takes several seconds.

The previous default was `Qwen/Qwen2.5-3B-Instruct` (about 6 GB, slower, usually a closer voice). Set `HF_MODEL` to that id to switch back. Both ids are named in `model_defaults.py`.

## Quick start

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open [http://localhost:3000](http://localhost:3000).

Optional settings live in `.env`. Copy `config/.env.example` if you want to change the model or port. `HF_TOKEN` is not required for Qwen.

`data/nick_valentine_dialogue.csv` is a short original sample so the app has something to load. It is not game script. Replace that file with your own lines — same columns, especially `RESPONSE TEXT`, `CONTEXT`, and `EMOTION` — then build the index the app reads:

```bash
python build_vector_db.py --csv data/nick_valentine_dialogue.csv
```

That writes `chroma_db_nick_valentine/`, which is gitignored. If the folder is missing, replies still work and example retrieval falls back to random CSV lines. Do not commit dialogue extracted from a commercial game.

Column details: [docs/BUILDING_VECTOR_DB.md](docs/BUILDING_VECTOR_DB.md).

## What you can do in the UI

- Pick a situation (investigation, combat, casual, emotional, and others) and a tone.
- Keep **Include examples** on to pull similar Nick lines into the prompt.
- The model sees the last few exchanges, so short follow-ups stay on topic.
- Reset or export the current conversation.
- The agent dropdown only lists models that actually started. Today that is the local Hugging Face agent.

## How a reply is built

1. `AgentManager` sends the request to the active agent.
2. The agent writes a system prompt from the chosen context and emotion, plus recent turns.
3. If examples are enabled, ChromaDB returns nearby dialogue lines. If that fails, a few CSV lines are used instead.
4. Qwen generates up to about two sentences. Obvious prompt leakage is stripped before the reply is shown.

Generation settings are fixed in `agents/huggingface_agent.py`: `temperature` 0.7, `top_p` 0.9, `top_k` 50, `max_new_tokens` 80.

## Project layout

```
app.py                     Flask routes and agent startup
agents/
  base_agent.py            Shared agent interface and history
  agent_manager.py         Registers agents and routes requests
  huggingface_agent.py     Qwen generation (used by the web app)
  gemini_agent.py          Optional Gemini client for compare_models.py only
build_vector_db.py         Build a ChromaDB index from a dialogue CSV
compare_models.py          Side-by-side local model vs Gemini script
dialogue_processor.py      CSV loader and retrieval labels
model_defaults.py          Current and previous Hugging Face model ids
vector_store.py            ChromaDB search. Expects chroma_db_nick_valentine/
nick_personality.py        Unused character profile (not wired into prompts)
data/                      Sample dialogue CSV. Replace this with your own file
public/                    Static UI
tests/test_smoke.py        Route tests with a fake agent. No model download
config/.env.example        Environment template
docs/                      Longer notes that do not belong in this file
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/generate` | Body: `userInput`, optional `context`, `emotion`, `includeExamples` |
| `GET` | `/api/history` | Active agent's history |
| `POST` | `/api/reset` | Clear that history |
| `GET` | `/api/export` | History plus model name |
| `GET` | `/api/agents` | Registered agents |
| `POST` | `/api/agents/switch` | Body: `agentId`. Resets that agent's history |
| `GET` | `/api/agents/active` | Active agent info |
| `GET` | `/api/health` | Process is up; check `agents_count` |
| `GET` | `/api/cost-estimate` | Always `$0.00` for the local model |

## Configuration

| Variable | Default | Used by |
| --- | --- | --- |
| `PORT` | `3000` | `app.py` |
| `HF_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Web app |
| `HF_TOKEN` | unset | Optional Hugging Face download auth |
| `GEMINI_API_KEY` | unset | `compare_models.py` only |
| `GEMINI_MODEL` | `gemini-1.5-flash` | `compare_models.py` only |

## Docs

- [Building the vector database](docs/BUILDING_VECTOR_DB.md)
- [Why a small local model was chosen over a general LLM](docs/MODEL_COMPARISON.md)

## Limits

- One character, one local model in the web UI. Gemini is a comparison script, not a second chat backend.
- The Chroma index is generated. Rebuild it after you change the CSV. It is not stored in git.
- `nick_personality.py` is not imported anywhere. The prompt text lives in the agent classes.
- Smoke tests do not load the model: `python -m unittest tests.test_smoke`
- The server binds `0.0.0.0` with Flask debug mode on. Fine for a local demo, not for a public host.

## License

Code and the sample CSV are under the MIT license in `LICENSE`.

That license does not cover dialogue you add yourself. If you drop in lines from a game, keep that file private and out of git.

# ianvex — VEX V5 Engineering Notebook Evaluator

An AI agent that reviews a VEX Robotics V5 **Engineering Notebook** and gives:

- Comments on **what to change / improve**
- Ideas for **what to add**
- A rubric-by-rubric evaluation

Every judgment is **grounded in the official VEX Engineering Notebook Rubric and
the current season "Override" game manual** — the AI is instructed never to
invent rubric criteria or game rules.

## Setup (one time)

1. **Install Python packages**
   ```
   pip install -r requirements.txt
   ```

2. **Add your API key**
   - Copy `.env.example` to `.env`
   - Paste your Anthropic API key into it.

3. **Add the reference documents** to the `reference/` folder:
   - `rubric.pdf` — the official VEX Engineering Notebook Rubric
   - `game_manual.pdf` — the Override game manual
   (You can also upload these from the app sidebar if you prefer.)

## Run

```
streamlit run app.py
```

A browser tab opens. Upload the notebook (PDF or images), optionally add a focus
note, and click **Evaluate notebook**.

## How it works

The app sends three things to Claude: the rubric, the game manual, and the
uploaded notebook. A strict system prompt forces the model to base all feedback
on those documents only, and to say so when something is not covered rather than
guessing.

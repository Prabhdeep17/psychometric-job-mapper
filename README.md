# Psychometric Job Mapper

Enter a job title and get the psychometric domains that matter most for
that role — ranked, with a short explanation of what each domain means
and why it fits the job.

Built on O*NET 31.0 data (1,016 occupations) mapped to a catalogue of
50 psychometric domains. Explanations are written by Groq (optional);
without an API key the app uses a built-in fallback, so it works either way.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional: live AI explanations

Copy `.env.example` to `.env` and add your Groq API key:

```
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
GROQ_API_KEY=your-key-here
```

On Streamlit Cloud, add the same values under **Settings → Secrets**.

## Data

`data/processed/` holds the occupation–domain mapping;
`data/raw/` holds the three O*NET reference tables the app reads at runtime.
Source: O*NET 31.0, U.S. Department of Labor (CC BY 4.0).

Scores are preliminary, occupation-relative indicators. The domain
mappings are pending expert review and must not be used for hiring decisions.

# Psychometric domain mapping

Type a job role. The app lists the stored work domains that fit that role.

Live app: https://prabhdeep17-psychometric-job-mapper-app-0oq174.streamlit.app/

## What a reply contains

For each domain the reply shows:

- the domain name
- the stored score out of 5
- what the domain means
- why it fits that role

The score and the definition come from the dataset. They are not rewritten. The reason uses the role’s stored work and that domain’s definition. When a Groq key is available, Groq writes that reason. When Groq is unavailable, the app still shows the domains from the stored role description. It does not invent a score.

## Which roles work

These 20 roles are stored. Each one has the same 50 domains, with its own scores.

1. Customer Support Specialist
2. Cybersecurity Analyst
3. Data Analyst
4. Data Scientist
5. Entrepreneur
6. Financial Analyst
7. HR Manager
8. Healthcare Professional
9. Lawyer
10. Machine Learning Engineer
11. Marketing Manager
12. Mechanical Engineer
13. Operations Manager
14. Product Manager
15. Project Manager
16. Research Scientist
17. Sales Representative
18. Software Engineer
19. Teacher
20. UX Designer

A few other titles are accepted as stand-ins. For example, “ML Engineer” uses Machine Learning Engineer. The reply says it is using that stored role. A title that is not one of the 20, and is not one of those stand-ins, does not get invented domains.

## How a domain is chosen

Every role already has a stored relevance score from 1 to 5 for every domain. The app does not calculate a new score.

| Score | Meaning |
| --- | --- |
| 5 | Core / highly salient |
| 4 | Highly relevant |
| 3 | Moderately relevant |
| 2 | Helpful but secondary |
| 1 | Low / indirect |

The chat shows the domains at that role’s highest score. For most roles that is 5. If fewer than five domains have that top score, the next score down is included as well.

Equal scores are ties. They are shown together. The order inside a tie is not a second ranking.

## What the score is

The 1–5 figure is a provisional role-level estimate stored in the dataset. It is not a personality test, not a mental-health result, and not a score of whether a person should be hired.

## Files

The app is small on purpose. Each file below is required. `app.py` is only the screen. The work of matching a role, reading scores, and writing the reason lives in `src/`. Removing `src/` stops the app.

| File | What it does |
| --- | --- |
| `app.py` | Opens the chat page. It shows the 20 roles, takes the title you type, and prints the reply. |
| `requirements.txt` | Lists the Python packages to install: Streamlit, pandas, httpx, and python-dotenv. |
| `data/psychometric_role_domain_mapping_corrected_master.csv` | The dataset. It holds 20 roles, 50 domains, each domain’s definition, the stored 1–5 score, and the role’s scope of work. This is the source of every score and definition. |
| `src/__init__.py` | Marks `src` as a Python package so the other files can be imported. |
| `src/config.py` | Reads settings such as `GROQ_API_KEY` from the environment. It does not contain a key. |
| `src/data_loader.py` | Loads the CSV, checks that the expected columns are present, and gives the rest of the app one role or one domain at a time. |
| `src/role_matcher.py` | Turns the typed title into one stored role. It accepts an exact name, a known stand-in such as “ML Engineer,” or a close title. If nothing matches, it does not invent a role. |
| `src/ranking_engine.py` | Sorts that role’s 50 domains by the stored 1–5 score. Equal scores stay tied. It never changes a score. |
| `src/chat_service.py` | Builds the reply: domain name, score, what the domain means, and why it fits. The score and the definition are copied from the CSV. |
| `src/groq_router.py` | Sends the role and the domain definitions to Groq so the “why it fits” line can be written. If Groq rejects the key or is unavailable, this file fails safely and the chat still shows the stored domains. |
| `.gitignore` | Keeps `.env` and local cache files out of the repository so the API key is not uploaded. |

## Run it

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Groq key

The key is not in this repository. For a local run, create a `.env` file with:

```bash
GROQ_API_KEY=your-key
```

On Streamlit Community Cloud, put `GROQ_API_KEY` in the app secrets. Do not commit the key.

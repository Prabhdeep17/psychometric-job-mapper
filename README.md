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

| File | Why it is here |
| --- | --- |
| `app.py` | The chat screen |
| `requirements.txt` | Packages needed to run the app |
| `data/psychometric_role_domain_mapping_corrected_master.csv` | The 20 roles, 50 domains, definitions, and 1–5 scores |
| `src/role_matcher.py` | Matches the typed title to a stored role |
| `src/data_loader.py` | Reads the CSV |
| `src/ranking_engine.py` | Orders domains by the stored score |
| `src/chat_service.py` | Builds the reply: score, meaning, and why it fits |
| `src/groq_router.py` | Calls Groq for the reason when a key is set |
| `src/config.py` | Reads settings from the environment |

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

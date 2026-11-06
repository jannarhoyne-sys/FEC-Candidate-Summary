## Predicting U.S. House Vote Share from Campaign Finance and Race Characteristics

An end-to-end machine learning model: from raw election and campaign finance data to a model deployed on Heroku, Docker Hub, AWS EC2, AWS SageMaker and Kubernetes.

**Live app:** https://election-2024-hoyne-5718a83521da.herokuapp.com

| Part | Deliverable | Link / location |
|---|---|---|
| 1 | Problem, data and features (this README) | `README.md` |
| 2 | Notebook and trained model | `Notebooks/project.ipynb`, `Models/vote_share_model.pkl` |
| 3 | Flask app on localhost | `app.py`, `templates/index.html`, screenshots in `Screenshots/` |
| 4 | CI/CD with GitHub Actions → Heroku | `.github/workflows/deploy.yml` · https://election-2024-hoyne-5718a83521da.herokuapp.com |
| 5 | Container on Docker Hub | `Dockerfile` · https://hub.docker.com/r/jannahoyne1/house-vote-share |
| 6 | Container on Heroku | `.github/workflows/deploy-container.yml` · https://election-2024-hoyne-docker-791a82f4180f.herokuapp.com |
| 7 | AWS EC2 | http://3.145.216.176 · screenshots in `Screenshots/` |
| 8 | AWS SageMaker | `sagemaker/deploy.py`, `sagemaker/sample_prediction.json` |
| — | Kubernetes configuration | `k8s/deployment.yaml`, `k8s/service.yaml` |

---

### 1. The problem

**Question:** How much of a candidate's 2024 U.S. House vote share can be explained by their fundraising compared with their opponents', together with party, incumbency and state?

Campaign money and election results are closely linked, but the relationship isn't simple. A candidate who raises twice as much as their opponent doesn't get twice the votes, partisanship sets a floor and a ceiling, and incumbents both raise more and win more. I wanted to measure how much money actually explains, and where it stops explaining.

**Why machine learning:** vote share depends on several factors at once (money relative to opponents, party, incumbency, number of candidates), and they interact. A tree-based model can learn those combinations and non-linear patterns directly from the data, instead of me guessing a formula. Cross-validation also gives an honest estimate of how well the model does on races it has never seen.

**Target:** each candidate's **vote share** = candidate's votes ÷ all valid votes in the district. The predicted winner of each race is the candidate with the highest predicted share.

### 2. Data

| Dataset | What it contains | Where to get it |
|---|---|---|
| **FEC Candidate Summary, 2023–2024** (`candidate_summary_2024.csv`) | Fundraising and spending totals for every federal candidate: receipts, individual, PAC, party and self-funding, cash on hand, debt | Federal Election Commission: https://www.fec.gov/data/browse-data/?tab=bulk-data |
| **MIT Election Data and Science Lab, U.S. House 1976–2024** (v15.0) (`1976-2024-house.tab`) | Official vote counts for every House candidate | Harvard Dataverse: https://doi.org/10.7910/DVN/IG0UN2 |
| FEC "All candidates" file, 2023–2024 (`weball24.txt`) | A second FEC summary file | FEC bulk data (link above). Checked as a backup for missing candidates, but not used |

The two main files share no ID, so I matched them by district and last name (see the notebook for the full matching process).

### Features

| Feature | Meaning |
|---|---|
| `state_po` | State |
| `party_group` | DEM, REP or OTHER |
| `status` | INCUMBENT, CHALLENGER or OPEN (open seat) |
| `n_candidates` | Number of candidates in the race |
| `money_share` | The candidate's money ÷ all money raised in the district |
| `money_gap` | The candidate's money vs. their top opponent's, on a log scale (+1 = raised 10× more) |
| `log_receipts` | Total raised, on a log scale ($100K = 5, $1M = 6) |
| `log_opp_money` | Opponents' combined money, on a log scale |
| `small_donor_pct` | Share of money from small (unitemized) donors |
| `pac_pct` | Share of money from PACs |
| `party_money_pct` | Share of money from party committees |
| `self_fund_pct` | Share of money from the candidate's own contributions and loans |
| `log_cash_start` | Cash on hand at the start of the cycle, on a log scale |

### Cleaning (full details in the notebook)

**FEC cleaning**
- Started with 5,302 candidates in the FEC file
- Kept only House candidates: 3,078
- Kept candidates who raised $5,000 or more: 1,722
  - The FEC requires candidates to register once they pass $5,000, so those below it usually never ran a real campaign
- For matching to the election results, all 3,078 House filers were used, so minor candidates who raised less than $5,000 still kept their real (small) fundraising totals

**MIT cleaning (2024 general election)**
- Started with 1,353 rows for 2024
- Removed blank, void, under-vote, over-vote and exhausted ballots: 1,284
  - These are ballots that didn't count as a vote for anyone
- Removed write-ins and rows that weren't real candidates (like "SCATTERING" and "ALL OTHERS"): 1,104
- Combined fusion-ticket rows: 1,069
  - In states like New York, one candidate can appear on several party lines, so their votes were added together into one row
- Dropped 22 uncontested races: 1,047 candidates in 414 districts
  - A candidate running unopposed gets 100%, which says nothing about money

**Vote share = candidate's votes ÷ all valid votes in the district**
- I calculated this myself instead of using MIT's `totalvotes` column, because New York's totals leave out blank ballots and Florida reports -1 for unopposed races

**Matching the two files**
- Started with 1,047 candidates from the MIT file
- Matched 1,032 on district + last name
  - 3 FEC records were fixed by hand: one listed under a married name, two with first and last names reversed
- Matched 1 more on state + last name (FEC had no district listed)
- 14 candidates were still unmatched:
  - Dropped the 9 districts with a missing major candidate (5% or more of the vote), since their money share couldn't be calculated
  - Kept the 4 minor candidates (under 5%) with $0, since they likely never had to file with the FEC

**Final dataset: 1,023 candidates in 405 districts**

### 3. Results

**Key findings from the data**
- The top fundraiser won **94%** of the 405 races, but money is far more lopsided than votes: candidates with almost no money still often got 30–45%, and candidates with nearly all the money usually got 55–85%.
- Incumbents had a median vote share of about 63%; challengers about 30%.
- Democrats and Republicans had nearly identical vote shares; party mainly separates major-party from third-party candidates.

**Models** (5-fold cross-validation, grouped by district so candidates from the same race never appear in both training and testing)

| Model | MAE (points) | R² |
|---|---|---|
| Random forest (**chosen**) | 4.08 ± 0.43 | 0.943 |
| Gradient boosting | 4.19 ± 0.38 | 0.942 |
| Linear regression | 4.83 ± 0.48 | 0.925 |
| Baseline (guess the average, validation set) | 21.0 | −0.01 |

**Final test set** (15% of districts, used once): **MAE 3.8 percentage points, R² 0.932, winner picked correctly in 93% of districts** (the top fundraiser alone: 90%).

**Most important features** (permutation importance): party, then the three money features (which share credit with each other), then number of candidates. Incumbency scored low on its own because the money features already carry that information.

**Limitations**
- **Close races:** in test districts decided by less than 10 points, the model picked 6 of 9 winners. With about 4.5 points of error per candidate, close races can easily flip.
- **Same-party races:** in top-two (California, Washington) and open-primary (Louisiana, Alaska) races, two candidates from the same party behave very differently from normal races. I tested two extra features for this (Version 2 in the notebook); they didn't help, so the app warns users on these races instead.
- **Correlation, not causation:** donors give to candidates who already look likely to win, so money is partly a *result* of expected success.
- **One election year:** the model has no feature for the national political environment, so it would likely miss systematically in a wave year.

### 4. The app

A Flask app (`app.py`) with two modes:
- **Look up a race:** pick any 2024 district and see each candidate's predicted vs. actual vote share, a race rating ("Safe," "Leans," "Too close to call") and whether the model called the winner. Each race can be opened as a what-if.
- **Hypothetical candidate:** enter 2–4 candidates (party, status, money raised, optional funding mix) and get predicted vote shares. Blank optional fields are filled with typical values for that party and status (`Models/defaults.json`).
- Every input has a short explanation underneath (what it means and what values it takes), and gray example values show the expected format.

It also has `/ping` and `/invocations` routes for Sagemaker.

**Run it locally**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Then open http://localhost:8080.

**Run it with Docker**
```
docker pull jannahoyne1/house-vote-share
docker run --rm -p 8080:8080 jannahoyne1/house-vote-share
```

### 5. Deployments

| Platform | How |
|---|---|
| **Heroku (Part 4)** | GitHub Actions installs the packages, tests that the app loads and its pages respond, then deploys with the `Procfile` (gunicorn). |
| **Docker Hub (Part 5)** | `Dockerfile` based on `python:3.12-slim`; `.dockerignore` keeps the image small (about 150 MB). |
| **Heroku container (Part 6)** | A second workflow builds the image, starts it, checks `/ping` and the home page, sets the Heroku app to the container stack, and deploys the container. |
| **AWS EC2 (Part 7)** | Amazon Linux 2023 (t3.micro); installed Docker and ran the Docker Hub image with port 80 mapped to the app's port 8080. |
| **AWS SageMaker (Part 8)** | Pushed the image to Amazon ECR; `sagemaker/deploy.py` creates a serverless endpoint and sends a sample race. Result: a $2M Democratic incumbent vs. a $500K Republican challenger → 64.1% vs. 35.9%, "Safe DEM" (`sagemaker/sample_prediction.json`). The endpoint was deleted afterward with `sagemaker/cleanup.py`. |
| **Kubernetes** | Local minikube cluster: `k8s/deployment.yaml` runs 2 replicas of the Docker Hub image with `/ping` health checks; `k8s/service.yaml` exposes it as a NodePort service. Applied with `kubectl apply -f k8s/`. |

## 6. Troubleshooting log

- **Heroku: "No Python version found in .python-version."** The file was empty. Adding `3.12` fixed it; the CI tests had already passed, showing the app itself was fine.
- **Docker: gunicorn `WORKER TIMEOUT … no URI read`.** The default "sync" worker timed out on idle browser connections. Switching to threaded workers (`--workers 2 --threads 4 --timeout 120`) fixed it.
- **Docker: "port is already allocated."** An older container was still running; stopped it with `docker ps` and `docker stop`.
- **EC2: browser timed out.** The container answered on the server (`curl localhost`) and from my computer (`curl.exe http://3.145.216.176`), which narrowed the problem to the security group. It needed an inbound rule allowing HTTP (port 80). The app serves `http://` only, so the address must be typed with `http://`.
- **SageMaker: AccessDeniedException** creating the ECR repository. The IAM user was missing `AmazonEC2ContainerRegistryFullAccess`.
- **SageMaker: "Unsupported manifest media type application/vnd.oci.image.manifest.v1+json."** Docker Desktop saves images in the newer OCI format, which Docker Hub, Heroku and EC2 accept but SageMaker doesn't. Rebuilt and pushed with `oci-mediatypes=false` to use the Docker v2 format.
- **SageMaker container start-up:** SageMaker starts containers with the argument `serve`, so the Dockerfile defines a small `serve` start script and uses `CMD ["serve"]`, which also works for Heroku and local runs.
- **Kubernetes:** ran minikube natively on Windows with the Docker driver instead of inside Ubuntu (WSL). It's the same single-node cluster and the same kubectl commands.

### 7. Monitoring and next steps

- **Ground truth:** the 2026 midterm results can be compared with this model's predictions for the same districts, a direct test of how a 2024-trained model holds up two years later.
- **Data drift:** 2026 is a midterm year with a different national environment. Comparing the 2026 inputs with the training data (Kolmogorov–Smirnov tests for numeric features, chi-squared for categorical ones) would show how far the data has shifted.
- **Retraining:** with more election years, the model could add a feature for the national environment and the primary system (top-two, open or partisan primary), which would address its two main weaknesses.

### Repository structure
```
├── app.py                       Flask app
├── templates/index.html         App page
├── Models/                      vote_share_model.pkl, defaults.json
├── Data/                        Raw and processed data
├── Notebooks/project.ipynb      Cleaning, EDA, features, models, evaluation
├── Figures/                     Charts from the notebook
├── Screenshots/                 Deployment screenshots
├── requirements.txt, Procfile, .python-version   Heroku (Part 4)
├── Dockerfile, .dockerignore    Container (Parts 5–8)
├── .github/workflows/           CI/CD: deploy.yml (Part 4), deploy-container.yml
├── sagemaker/                   deploy.py, cleanup.py, sample_prediction.json
└── k8s/                         deployment.yaml, service.yaml (Kubernetes)
```

### References
- Federal Election Commission. (2024). *Candidate summary, 2023–2024* [Data set]. https://www.fec.gov/data/browse-data/?tab=bulk-data
- MIT Election Data and Science Lab. (2017). *U.S. House 1976–2024* (Version 15.0) [Data set]. Harvard Dataverse. https://doi.org/10.7910/DVN/IG0UN2

### Design credit
The app's results layout (the vote-share bar with a 50% marker, large predicted percentages and striped "too close to call" shading) was inspired by the Associated Press's 2026 midterm election results coverage: https://apnews.com/projects/elections-2026/
 This project is not affiliated with or endorsed by AP.
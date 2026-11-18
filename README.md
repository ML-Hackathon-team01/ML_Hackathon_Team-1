# PathSnap — Distributed Map-Matching & Route Reconstruction (Team-01)

**Course:** UE24CS352A: Machine Learning (PES University, RR Campus)  
**Team-01:** Navya (Logistic Regression, SVM), Neal (Decision Tree), Shreyas (Random Forest), Sohel (Gradient Boosting)

---

## Repository Structure

```text
ML_Hackathon_Team-1/
├── data_inspection/         # Raw file loading & missing-value audit
├── preprocessing/           # Cleaning, BallTree candidate generation & feature preparation
├── models/                  # Model training scripts (all 5 models + standalone Gradient Boosting)
├── Model evaluation/        # Evaluation metrics, comparison tables, plots, confusion matrices & Folium maps
├── saved_models/            # Compact trained final model artifact (.joblib)
├── api/                     # Laptop 1: FastAPI backend, model-loading logic, HMM/Viterbi & server-side evaluation
├── ui/                      # Laptop 2: Evaluator-facing Streamlit web interface
├── deployment/              # Config file & one-click startup scripts for Laptop 1 and Laptop 2
├── results/                 # Cleaning reports & intermediate outputs
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 1. Installation

Use **Python 3.12** (recommended):

```powershell
git clone https://github.com/ML-Hackathon-team01/ML_Hackathon_Team-1.git
cd ML_Hackathon_Team-1
python -m pip install -r requirements.txt
```

Place the provided `Student folder` alongside the cloned repository (`../Student folder`) containing `train/`, `student_test/`, and (during final evaluation on Laptop 1 only) `evaluator_test/`.

---

## 2. Running Data Processing, Training & Evaluation

```powershell
# Part 1: Data inspection & preprocessing (80 training trips)
python data_inspection/check_missing_data.py
python preprocessing/prepare_dataset.py

# Part 2: Train & compare all 5 models (5-fold GroupKFold CV + Student-Test)
python models/train_models.py

# Optional: Run standalone Gradient Boosting (Sohel)
python models/sohel_gradient_boosting.py

# Generate Section 3.3 evaluation plots & Folium map
python "Model evaluation/generate_section3_3_figures.py"
```

---

## 3. Two-Laptop Deployment Instructions (Section 7)

### Architecture
- **Laptop 1 (`api/`):** Hosts the trained Gradient Boosting model (`saved_models/sohel_gradient_boosting.joblib`), road networks, HMM/Viterbi route reconstruction pipeline, and server-side evaluation. Ground-truth `.route` files remain strictly on Laptop 1.
- **Laptop 2 (`ui/`):** Runs the Evaluator UI, displays the 10 evaluation trip options fetched from Laptop 1, sends only `trip_id` to `POST /predict`, and renders the reconstructed route map and metrics.

### Step A — Start Laptop 1 (Model + FastAPI Backend)

Connect both laptops to the **same Wi-Fi or mobile hotspot**, then run on **Laptop 1**:

```powershell
# Optional: explicitly point to the reserved 10 evaluator trips folder
# $env:PATHSNAP_EVAL_DIR = "C:\path\to\evaluator_test"

.\deployment\start_laptop1_api.ps1
```
*(Or run directly: `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000`)*

Note the LAN URL printed in Laptop 1's terminal (for example, `http://192.168.1.25:8000`).  
Keep this CMD/PowerShell window open during evaluation to show live incoming API request logs and HTTP `200 OK` responses.

> **Windows Firewall Tip (if Laptop 2 cannot reach Laptop 1):** Allow TCP port `8000` on Laptop 1 once (in Admin PowerShell):
> ```powershell
> New-NetFirewallRule -DisplayName "PathSnap FastAPI 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
> ```

### Step B — Start Laptop 2 (Evaluator UI)

On **Laptop 2** (or a second terminal on the same laptop for local testing):

```powershell
.\deployment\start_laptop2_ui.ps1
```
*(Or run directly: `python -m streamlit run ui/app.py --server.port 8501`)*

1. Open the browser at `http://localhost:8501`.
2. In the left sidebar, set **Laptop 1 API URL** to `http://<LAPTOP1_IP>:8000` (or `http://127.0.0.1:8000` when testing on one laptop).
3. Confirm the green **API Online** badge appears and all **10 evaluation trips** load in the dropdown.
4. Select any trip and click **Reconstruct Route** to run live inference on Laptop 1 and visualize the reconstructed route & metrics.

### API Endpoints Reference

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns backend status, loaded model name, decoder, and active partition |
| `GET` | `/trips` | Lists the 10 available evaluation trips (IDs & GPS counts only) |
| `POST` | `/predict` | Accepts `{"trip_id": "00000000"}`, runs live ML + HMM/Viterbi inference & server-side evaluation, returns reconstructed route & metrics |
| `GET` | `/evaluate/{trip_id}` | Convenience GET endpoint for quick testing via browser or `curl` |

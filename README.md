# Morocco Climate Shock & Macroeconomic Transmission Pipeline

An end-to-end data engineering pipeline that extracts regional historical time-series indicators, processes structural transformations within a localized PostgreSQL database environment, and evaluates predictive matrices using an ensemble machine learning algorithm to quantify how climate-induced agricultural shocks transmit into macroeconomic labor markets.

---

## 1. System Architecture Layer

The infrastructure manages data through a decoupled execution flow to maintain environment cleanliness and optimized processing speeds:

* **Ingestion & ETL Engine:** Implemented via a vectorized Jupyter Notebook framework managing extraction and linear forward-filling interpolation to eliminate trend distortion.
* **Data Warehouse Layer:** A relational PostgreSQL instances housing the structured, optimized analytics schema (`morocco_economic_trends`).
* **Analytics Interface:** A streamlined, enterprise-grade data product built natively in Streamlit for real-time model inference and trend tracking.

---

## 2. Technical Stack & Dependencies

To maintain a minimal structural footprint, the pipeline operates strictly on core production libraries without workspace bloat:
* **Database Engine:** PostgreSQL 16 / SQLAlchemy Connection Pooler
* **Data Processing:** Pandas / NumPy
* **Predictive Compute:** Scikit-Learn (Ensemble Random Forest Engine)
* **Interface Layer:** Streamlit Architecture

---

## 3. Data Schema & Provenance

The system integrates data spanning from 1990 to 2024, enforcing strict typed constraints within the PostgreSQL instance:

| Column Name | Database Type | Description / Metric Mapping |
| :--- | :--- | :--- |
| `Year` | `INT` (Primary Key) | Temporal tracking baseline |
| `Crop_Index` | `FLOAT` | Normalized Crop Production Index (Ecological Baseline) |
| `Unemployment_Rate` | `FLOAT` | National Unemployment Percentage (Target Variable) |
| `Agriculture_GDP_Share`| `FLOAT` | Percentage share of primary sector macro-contribution |
| `GDP_Growth` | `FLOAT` | Annual national economic growth performance indicator |

---

## 4. Machine Learning & Statistical Attribution

To move past simple observations, the pipeline executes a local **Random Forest Regressor** to evaluate structural weight distributions:
* **Target Vector ($y$):** National Unemployment Rate
* **Feature Matrices ($X$):** Crop Production Index, GDP Growth, Agricultural GDP Share
* **Objective:** Isolate relative feature importances to determine if labor market stability is mathematically tied to primary sector climate conditions rather than generic secondary financial services.

---

## 5. Local Infrastructure Execution Guide

To initialize and deploy the pipeline locally on your machine, follow these precise environment protocols:

### Prerequisites
Ensure your local PostgreSQL server is active and housing a database named `world_economics` with your target data loaded into `morocco_economic_trends`.

### Execution Commands

1. **Sync your terminal directory to the repository path:**
   ```bash
   cd "$(dirname "$(find . -name app.py -print -quit)")"    
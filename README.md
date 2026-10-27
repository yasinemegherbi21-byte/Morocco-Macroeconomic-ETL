# Rain and the Economy – Morocco Macroeconomic ETL

A large part of Morocco's economy depends on farming, and farming depends on rain. A dry year hurts crop output, then GDP, then jobs. I wanted to measure that chain with real data instead of just reading about it.

This repo has two parts:

1. **A notebook** (`morocco_agriculture.ipynb`) that pulls four World Bank indicators for Morocco (1990–2024) through the API, cleans them with Pandas, stores them in PostgreSQL and then tries, step by step, to predict unemployment from crop data.
2. **A PySpark loader** (`etl_pipeline.py`) that reads the World Bank CSV exports (unemployment, GDP, inflation), standardises the columns and writes them to PostgreSQL through JDBC. I wrote it to learn how the same load looks in Spark.

A small Streamlit page (`app.py`) reads the PostgreSQL table and shows the crop index against unemployment, plus which variable a Random Forest leans on.

## What the data says

- The crop production index and GDP growth move together almost year for year. The 1995 drought pushed GDP growth to about -5%; 2020 combined a dry year with COVID.
- Crop index and unemployment move in opposite directions: low harvests (index around 40) line up with the highest unemployment (around 14%).
- Agriculture's weight in GDP changes from year to year with the harvest, which is exactly why a dry year is felt by the whole economy.

## What I learned trying to predict unemployment

I went through five modelling steps on a chronological split (train 1990–2017, test 2018–2024):

| Step | Model | Test R² |
|---|---|---|
| Baseline | Linear regression on crop index only | -2.27 |
| + lag features | Linear regression with last year's crop index and GDP growth | -3.05 |
| + agriculture share | Linear regression, four features | -2.36 |
| Random Forest | Same four features | -0.19 |
| 5-fold cross-validation | Random Forest | -10.70 on average |

Every score is negative, which means none of these models beats simply predicting the average. With 35 yearly rows, a 7-year test window and a target that depends on much more than harvests, that is the honest result. The value of the project is the pipeline and the analysis, not a forecast, and I preferred to show the real numbers rather than a fitted-on-everything score.

## Stack

Python · Pandas · PySpark · PostgreSQL · SQLAlchemy · Scikit-learn · Statsmodels · Matplotlib / Seaborn · Streamlit

## Run it

You need PostgreSQL running locally with a database named `world_economics`.

```bash
git clone https://github.com/yasinemegherbi21-byte/Morocco-Macroeconomic-ETL.git
cd Morocco-Macroeconomic-ETL
pip install -r requirements.txt
cp .env.example .env        # set DB_PASSWORD
```

**Notebook path (API → Pandas → PostgreSQL → analysis):**

```bash
jupyter notebook morocco_agriculture.ipynb
```

Run the cells top to bottom. The download cell creates `morocco_data/morocco_macro_clean.csv`; the PostgreSQL cell creates the table `morocco_economic_trends`.

**PySpark path (CSV → Spark → PostgreSQL):**

1. Download the three indicator CSVs from the World Bank (unemployment `SL.UEM.TOTL.ZS`, GDP `NY.GDP.MKTP.CD`, inflation `FP.CPI.TOTL.ZG`) into `data/`.
2. Download the PostgreSQL JDBC driver (`postgresql-42.7.3.jar`) next to the script, or set `JDBC_JAR` in `.env`.
3. Java 17+ must be installed.

```bash
python etl_pipeline.py
```

**Dashboard:**

```bash
streamlit run app.py
```

## Table

`morocco_economic_trends`, one row per year from 1990 to 2024:

| Column | Type | Meaning |
|---|---|---|
| `Year` | int | year |
| `Crop_Index` | float | World Bank crop production index (`AG.PRD.CROP.XD`) |
| `GDP_Growth` | float | annual GDP growth, % (`NY.GDP.MKTP.KD.ZG`) |
| `Unemployment_Rate` | float | national unemployment, % (`SL.UEM.TOTL.ZS`) |
| `Agriculture_GDP_Share` | float | agriculture as % of GDP (`NV.AGR.TOTL.ZS`) |

Missing years are filled by linear interpolation; the handful of gaps at the edges are back/forward filled.

## Limitations

- 35 yearly observations is very little for any model.
- The PySpark loader and the notebook load different indicators into different tables; they were written at different times.
- `morocco_data/` and `data/` are not versioned; both are recreated by the steps above.

## Author

Yassine Megherbi – [LinkedIn](https://www.linkedin.com/in/yassine-megherbi-b7b277263/) · [GitHub](https://github.com/yasinemegherbi21-byte)

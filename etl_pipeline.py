import os
from pyspark.sql import SparkSession

# 1. MAC & JAVA FIXES
# This prevents the "InaccessibleObjectException" on macOS with Java 17+
os.environ['JDK_JAVA_OPTIONS'] = '--add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED'

# 2. INITIALIZE SPARK
spark = SparkSession.builder \
    .appName("WorldBank_ETL_Final") \
    .config("spark.jars", "postgresql-42.7.3.jar") \
    .config("spark.driver.extraJavaOptions", "--add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED") \
    .getOrCreate()

# 3. DEFINE THE EXACT FILENAMES FROM YOUR UPLOADS
unemployment_csv = "API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_115692.csv"
gdp_csv          = "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_126992.csv"
inflation_csv    = "API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_115367.csv"

def load_world_bank_data(file_path):
    """Manually skips the first 4 rows to ensure the real header is captured."""
    # 1. Read the file as text lines
    raw_rdd = spark.sparkContext.textFile(file_path)
    
    # 2. Filter out the first 4 rows (index 0, 1, 2, 3)
    # zipWithIndex creates (line_content, index)
    clean_rdd = raw_rdd.zipWithIndex() \
        .filter(lambda x: x[1] >= 4) \
        .map(lambda x: x[0])
    
    # 3. Convert the cleaned text back into a CSV DataFrame
    return spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(clean_rdd)

try:
    print("--- Extracting Data from CSVs ---")
    df_unemployment = load_world_bank_data(unemployment_csv)
    df_gdp          = load_world_bank_data(gdp_csv)
    df_inflation    = load_world_bank_data(inflation_csv)

    # 4. TRANSFORMATION: Standardize Column Names
    # PostgreSQL works best with lowercase and underscores (no spaces)
    def rename_cols(df):
        return df.withColumnRenamed("Country Name", "country_name") \
                 .withColumnRenamed("Country Code", "country_code") \
                 .withColumnRenamed("Indicator Name", "indicator_name") \
                 .withColumnRenamed("Indicator Code", "indicator_code")

    df_unemployment = rename_cols(df_unemployment)
    df_gdp          = rename_cols(df_gdp)
    df_inflation    = rename_cols(df_inflation)

    # 5. LOADING TO POSTGRESQL
    db_url = "jdbc:postgresql://localhost:5432/world_economics"
    db_properties = {
        "user": "postgres",
        "password": "7777",  # Ensure this is your actual password
        "driver": "org.postgresql.Driver"
    }

    print("--- Loading to PostgreSQL: table 'unemployment' ---")
    df_unemployment.write.jdbc(url=db_url, table="unemployment", mode="overwrite", properties=db_properties)

    print("--- Loading to PostgreSQL: table 'gdp' ---")
    df_gdp.write.jdbc(url=db_url, table="gdp", mode="overwrite", properties=db_properties)

    print("--- Loading to PostgreSQL: table 'inflation' ---")
    df_inflation.write.jdbc(url=db_url, table="inflation", mode="overwrite", properties=db_properties)

    print("\n--- ETL PROCESS COMPLETE SUCCESSFULY ---")

except Exception as e:
    print(f"\n--- ERROR --- \n{e}")

finally:
    spark.stop()
    print("--- Spark Session Closed ---")

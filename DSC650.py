import happybase
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("HospitalReadmission30DayRF")
    .enableHiveSupport()
    .getOrCreate()
)

df = spark.sql("""
    SELECT
        race,
        gender,
        age,
        time_in_hospital,
        num_lab_procedures,
        num_procedures,
        num_medications,
        number_outpatient,
        number_emergency,
        number_inpatient,
        number_diagnoses,
        readmitted
    FROM healthcare_db.hospital_readmissions
    WHERE readmitted IN ('<30', '>30', 'NO')
""")

# Clean '?' placeholders and define binary target: 1.0 = readmitted <30 days, 0.0 = otherwise
clean_df = (
    df.withColumn("race", F.when(F.col("race") == "?", "Unknown").otherwise(F.col("race")))
      .withColumn("gender", F.when(F.col("gender") == "?", "Unknown").otherwise(F.col("gender")))
      .withColumn("label", F.when(F.col("readmitted") == "<30", 1.0).otherwise(0.0))
      .dropna(subset=[
          "time_in_hospital", "num_lab_procedures", "num_procedures",
          "num_medications", "number_outpatient", "number_emergency",
          "number_inpatient", "number_diagnoses", "label"
      ])
)

cat_cols = ["race", "gender", "age"]
num_cols = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"
]

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in cat_cols
]

assembler = VectorAssembler(
    inputCols=[f"{c}_idx" for c in cat_cols] + num_cols,
    outputCol="features",
    handleInvalid="skip"
)

rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=50,
    maxDepth=8,
    seed=42
)

pipeline = Pipeline(stages=indexers + [assembler, rf])

train_df, test_df = clean_df.randomSplit([0.75, 0.25], seed=42)
model = pipeline.fit(train_df)
predictions = model.transform(test_df)

auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

auc_score = auc_eval.evaluate(predictions)
accuracy = acc_eval.evaluate(predictions)
f1_score = f1_eval.evaluate(predictions)

print(f"ROC-AUC: {auc_score:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1_score:.4f}")

metrics_rows = [
    ("rf_readmission_v1", "metrics:roc_auc", f"{auc_score:.4f}"),
    ("rf_readmission_v1", "metrics:accuracy", f"{accuracy:.4f}"),
    ("rf_readmission_v1", "metrics:f1", f"{f1_score:.4f}"),
]

def write_partition_to_hbase(partition):
    conn = happybase.Connection("master")
    conn.open()
    table = conn.table("readmission_metrics")
    for row_key, col_qual, val in partition:
        table.put(row_key, {col_qual: val})
    conn.close()

spark.sparkContext.parallelize(metrics_rows).foreachPartition(write_partition_to_hbase)
spark.stop()
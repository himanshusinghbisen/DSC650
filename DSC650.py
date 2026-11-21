import happybase
from pathlib import Path
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType, StringType

spark = (
    SparkSession.builder
    .appName("HospitalAdmissionsReadmissionML")
    .enableHiveSupport()
    .getOrCreate()
)

df = spark.sql("SELECT * FROM healthcare_db.hospital_readmissions")

# Optional: set exact target column name if known; otherwise defaults to the last column
preferred_targets = ["readmitted", "readmitted_30d", "readmission", "target", "label"]
target_col = next((c for c in preferred_targets if c in df.columns), df.columns[-1])

# Build clean numeric label (1.0 if string is '<30'/'1'/'yes'/'true', 0.0 if '0'/'no'/'false', or numeric > 0)
target_field = dict((f.name, f) for f in df.schema.fields)[target_col]

if isinstance(target_field.dataType, StringType):
    trimmed = F.lower(F.trim(F.col(target_col)))
    labeled_df = df.withColumn(
        "label",
        F.when(trimmed.isin("<30", "1", "yes", "y", "true", "readmitted"), F.lit(1.0))
         .when(trimmed.isin(">30", "no", "0", "n", "false"), F.lit(0.0))
         .otherwise(F.when(F.col(target_col).cast("double") > 0, F.lit(1.0)).otherwise(F.lit(0.0)))
    )
else:
    labeled_df = df.withColumn(
        "label",
        F.when(F.col(target_col).cast("double") > 0, F.lit(1.0)).otherwise(F.lit(0.0))
    )

labeled_df = labeled_df.filter(F.col("label").isin(0.0, 1.0))

feature_candidates = [c for c in df.columns if c != target_col]
numeric_cols = []
categorical_cols = []

for col_name in feature_candidates:
    dtype = dict((f.name, f) for f in df.schema.fields)[col_name].dataType
    if isinstance(dtype, (IntegerType, LongType, DoubleType, FloatType)):
        labeled_df = labeled_df.withColumn(col_name, F.col(col_name).cast("double"))
        numeric_cols.append(col_name)
    else:
        cleaned_str = (
            F.when(
                F.col(col_name).isNull()
                | (F.trim(F.col(col_name)) == "")
                | (F.col(col_name).isin("?", "NA", "N/A", "null", "NULL")),
                F.lit("Missing"),
            ).otherwise(F.trim(F.col(col_name)))
        )
        labeled_df = labeled_df.withColumn(col_name, cleaned_str)
        categorical_cols.append(col_name)

# Drop high-cardinality ID-like string columns (>100 distinct values) to keep trees fast on 16GB VM
filtered_cat_cols = []
for c in categorical_cols:
    distinct_cnt = labeled_df.select(c).distinct().count()
    if 1 < distinct_cnt <= 100:
        filtered_cat_cols.append(c)

labeled_df = labeled_df.dropna(subset=numeric_cols + ["label"])

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in filtered_cat_cols
]

assembler_inputs = [f"{c}_idx" for c in filtered_cat_cols] + numeric_cols
assembler = VectorAssembler(
    inputCols=assembler_inputs,
    outputCol="features",
    handleInvalid="skip",
)

rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=40,
    maxDepth=6,
    seed=42,
)

pipeline = Pipeline(stages=indexers + [assembler, rf])

train_df, test_df = labeled_df.randomSplit([0.75, 0.25], seed=42)
model = pipeline.fit(train_df)
predictions = model.transform(test_df)

roc_auc = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
).evaluate(predictions)
accuracy = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
).evaluate(predictions)
f1_score = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
).evaluate(predictions)

print(f"Target Column Used: {target_col}")
print(f"Features Used ({len(assembler_inputs)}): {assembler_inputs}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1_score:.4f}")

metrics_rows = [
    ("hospital_admissions_rf", "metrics:roc_auc", f"{roc_auc:.4f}"),
    ("hospital_admissions_rf", "metrics:accuracy", f"{accuracy:.4f}"),
    ("hospital_admissions_rf", "metrics:f1", f"{f1_score:.4f}"),
]

# Write the model results to the local filesystem alongside this script.
local_output_dir = Path(__file__).resolve().parent / "spark_output"
metrics_df = spark.createDataFrame(
    metrics_rows,
    ["model", "metric", "value"],
)
metrics_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(
    str(local_output_dir / "metrics")
)
predictions.select(target_col, "label", "prediction").coalesce(1).write.mode(
    "overwrite"
).option("header", "true").csv(str(local_output_dir / "predictions"))

print(f"Local metrics output: {local_output_dir / 'metrics'}")
print(f"Local predictions output: {local_output_dir / 'predictions'}")

def write_to_hbase_partition(partition):
    conn = happybase.Connection("master")
    conn.open()
    table = conn.table("readmission_metrics")
    for row_key, col_qual, val in partition:
        table.put(row_key, {col_qual: val})
    conn.close()

spark.sparkContext.parallelize(metrics_rows).foreachPartition(write_to_hbase_partition)
spark.stop()
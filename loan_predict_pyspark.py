import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, avg
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import (RandomForestClassifier, LogisticRegression, DecisionTreeClassifier)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Resolve base directory of this script
base_dir = os.path.abspath(os.path.dirname(__file__))

# Construct full input and output paths
input_path = f"file://{os.path.join(base_dir, 'loan_data.csv')}"
output_path = os.path.join(base_dir, "output.txt")

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("LoanPrediction").getOrCreate()

# Step 2: Load dataset
df = spark.read.csv(input_path, header=True, inferSchema=True)

# Step 3: Drop unnecessary columns - Loan_ID does not have any predictive value
df = df.drop("Loan_ID")

# Step 4: Handle Missing Values
# Numeric columns are imputed using the mean
numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                "Loan_Amount_Term", "Credit_History"]

impute_values = {}
for c in numeric_cols:
    mean_val = df.select(avg(col(c))).first()[0]
    impute_values[c] = mean_val

# Categorical values are filled with new category "Unknown"
categorical_cols = ["Gender", "Married", "Dependents", "Education",
                    "Self_Employed", "Property_Area", "Loan_Status"]

for c in categorical_cols:
    impute_values[c] = "Unknown"

df = df.fillna(impute_values)

# Step 5: Discretize required numeric columns
df = df.withColumn(
    "ApplicantIncome_bin",
    when(col("ApplicantIncome") < 2500, 0)
    .when(col("ApplicantIncome") < 5000, 1)
    .otherwise(2)
)

df = df.withColumn(
    "CoapplicantIncome_bin",
    when(col("CoapplicantIncome") == 0, 0)
    .when(col("CoapplicantIncome") < 1500, 1)
    .when(col("CoapplicantIncome") < 3000, 2)
    .otherwise(3)
)

df = df.withColumn(
    "LoanAmount_bin",
    when(col("LoanAmount") < 100, 0)
    .when(col("LoanAmount") < 200, 1)
    .otherwise(2)
)

df = df.withColumn(
    "Loan_Amount_Term_bin",
    when(col("Loan_Amount_Term") < 180, 0)
    .when(col("Loan_Amount_Term") <= 360, 1)
    .otherwise(2)
)

# Step 6: Encode categorical variables
indexers = [
    StringIndexer(inputCol="Gender", outputCol="GenderIndexed"),
    StringIndexer(inputCol="Married", outputCol="MarriedIndexed"),
    StringIndexer(inputCol="Dependents", outputCol="DependentsIndexed"),
    StringIndexer(inputCol="Education", outputCol="EducationIndexed"),
    StringIndexer(inputCol="Self_Employed", outputCol="SelfEmployedIndexed"),
    StringIndexer(inputCol="Property_Area", outputCol="PropertyAreaIndexed"),
    StringIndexer(inputCol="Loan_Status", outputCol="label")
]

# Step 7: Assemble features
assembler = VectorAssembler(
    inputCols=[
        "GenderIndexed",
        "MarriedIndexed",
        "DependentsIndexed",
        "EducationIndexed",
        "SelfEmployedIndexed",
        "PropertyAreaIndexed",
        "ApplicantIncome_bin",
        "CoapplicantIncome_bin",
        "LoanAmount_bin",
        "Loan_Amount_Term_bin",
        "Credit_History"
    ],
    outputCol="features"
)

# Step 8: Create three models
lr = LogisticRegression(featuresCol="features", labelCol="label")
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100)

# Step 9: Build pipelines
pipeline_lr = Pipeline(stages=indexers + [assembler, lr])
pipeline_dt = Pipeline(stages=indexers + [assembler, dt])
pipeline_rf = Pipeline(stages=indexers + [assembler, rf])

# Step 10: Train-test split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Step 11: Train models
model_lr = pipeline_lr.fit(train_data)
model_dt = pipeline_dt.fit(train_data)
model_rf = pipeline_rf.fit(train_data)

# Step 12: Predict
pred_lr = model_lr.transform(test_data)
pred_dt = model_dt.transform(test_data)
pred_rf = model_rf.transform(test_data)

# Step 13: Evaluate
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

acc_lr = evaluator.evaluate(pred_lr)
acc_dt = evaluator.evaluate(pred_dt)
acc_rf = evaluator.evaluate(pred_rf)

# Write accuracy to output file
with open(output_path, "w") as f:
    f.write(f"Logistic Regression Accuracy: {acc_lr:.4f}\n")
    f.write(f"Decision Tree Accuracy: {acc_dt:.4f}\n")
    f.write(f"Random Forest Accuracy: {acc_rf:.4f}\n")

# Stop Spark
spark.stop()

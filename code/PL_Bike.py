import pyspark.sql.types 
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler,VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql.functions import udf,col
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SQLContext

global Path    
Path="file:/home/u0416069/hw4/"
def CreateSparkContext():
    def SetLogger( sc ):
        logger = sc._jvm.org.apache.log4j
        logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
        logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
        logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)    

    sparkConf = SparkConf().setAppName("RunDecisionTreeBinary").set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    print(("master="+sc.master))    
    SetLogger(sc)
    return (sc)

sc = CreateSparkContext()
print("read data")
sqlContext = SQLContext(sc)
row_df = sqlContext.read.format("csv").option("header", "true").load(Path+"data/hour.csv")
#for column in row_df.columns:
#	print(column)

row_df = row_df.drop('instant').drop('dteday').drop('yr').drop('casual').drop('registered')
df = row_df.select([ col(l).cast("double").alias(l) for l in row_df.columns] )
train_df, test_df = df.randomSplit([0.7, 0.3])
train_df.cache()
test_df.cache()

print("setup pipeline")
assemblerInputs = df.columns[:-1]
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="t_features")
VectorIndexer = VectorIndexer(inputCol="t_features", outputCol="features", maxCategories=24)
dt = DecisionTreeRegressor(labelCol="cnt",  featuresCol="features", impurity="variance",maxDepth=10, maxBins=100)
pipeline = Pipeline(stages=[assembler, VectorIndexer,dt])

print("train model")
pipelineModel = pipeline.fit(train_df)
print("predict")
predicted=pipelineModel.transform(test_df)
selected_col = predicted.select('season','mnth','hr','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','cnt','prediction')
selected_col.show(10)
print("eval model")
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="cnt", metricName="rmse")
predictions = pipelineModel.transform(test_df)
auc = evaluator.evaluate(predictions)
print(auc)

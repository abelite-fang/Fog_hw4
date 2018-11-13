import pyspark.sql.types 
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import  StringIndexer, OneHotEncoder,VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql.functions import udf,col
from pyspark.ml.evaluation import BinaryClassificationEvaluator
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
row_df = sqlContext.read.format("csv").option("header", "true").option("delimiter", "\t").load(Path+"data/train.csv")
df= row_df.select(['url','alchemy_category' ]+[col(column).cast("double").alias(column) for column in row_df.columns[4:] ] )

train_df, test_df = df.randomSplit([0.7, 0.3])
train_df.cache()
test_df.cache()

print("setup pipeline")
dt = DecisionTreeClassifier(labelCol="label",  featuresCol="features", impurity="gini",maxDepth=10, maxBins=14)
stringIndexer = StringIndexer(inputCol='alchemy_category', outputCol="alchemy_category_Index")
encoder = OneHotEncoder(dropLast=False, inputCol='alchemy_category_Index', outputCol="alchemy_category_IndexVec")
assemblerInputs =['alchemy_category_IndexVec']  + row_df.columns[4:-1] 
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
pipeline = Pipeline(stages=[stringIndexer,encoder ,assembler,dt])

print("train model")
pipelineModel = pipeline.fit(train_df)
print("predict")
predicted=pipelineModel.transform(test_df).select('url','prediction').show(10)
print(predicted)
print("eval model")
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC"  )
predictions =pipelineModel.transform(test_df)
auc= evaluator.evaluate(predictions)
print(auc)

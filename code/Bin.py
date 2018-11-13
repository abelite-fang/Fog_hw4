from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.mllib.evaluation import BinaryClassificationMetrics

global Path
Path="file:/home/u0416069/hw4/"

def MkDataSource():
    def DataPreprocess(url):
        rawDataWithHeader = sc.textFile(url)
        header = rawDataWithHeader.first() 
        rawData = rawDataWithHeader.filter(lambda x:x !=header)    
        rData=rawData.map(lambda x: x.replace("\"", ""))    
        lines = rData.map(lambda x: x.split("\t"))
        return lines
    categoriesMap = []
    def extract_features(field,featureEnd):
        categoryIdx = categoriesMap[field[3]]
        categoryFeatures = np.zeros(len(categoriesMap))
        categoryFeatures[categoryIdx] = 1
        numericalFeatures=[field  for  field in field[4: featureEnd]]    
        return  np.concatenate(( categoryFeatures, numericalFeatures))
    def extract_label(record):
        label=(record[-1])
        return float(label)
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

    trainLine = []
    validLine = []
    def PrepareTrainData(): 
        nonlocal categoriesMap
        nonlocal trainLine
        nonlocal validLine
        lines = DataPreprocess(Path+"data/train.csv")

        categoriesMap = lines.map(lambda fields: fields[3]).distinct().zipWithIndex().collectAsMap()
        print(lines.map(lambda fields: fields).take(1))
        (trainLine, validLine) = lines.randomSplit([7, 3])
        trainData = trainLine.map( lambda r:LabeledPoint( extract_label(r), extract_features(r,len(r) - 1)))
        validationData = validLine.map( lambda r:LabeledPoint( extract_label(r), extract_features(r,len(r) - 1)))
        return (trainData, validationData)
    def PredictTestData(model): 
        nonlocal categoriesMap
        nonlocal validLine
        dataRDD = validLine.map(lambda r: ( r[0] ,extract_features(r,len(r) )))
        for data in dataRDD.take(10):
            predictResult = model.predict(data[1])
            print(" url：" +str(data[0][:20]) +".... ==>prediction:"+ str(predictResult))
    
    return (PrepareTrainData,PredictTestData)
    
def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels=score.zip(validationData.map(lambda p: p.label))
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    AUC=metrics.areaUnderROC
    return AUC

def trainEvaluateModel(trainData,validationData,
                                        impurityParm, maxDepthParm, maxBinsParm):
    model = DecisionTree.trainClassifier(trainData,
                numClasses=2, categoricalFeaturesInfo={},
                impurity=impurityParm,
                maxDepth=maxDepthParm, 
                maxBins=maxBinsParm)
    return model

if __name__ == "__main__":
    (PrepareData,PredictTestData) = MkDataSource()
    print("read data")
    (trainData, validationData) =PrepareData()
    trainData.persist(); validationData.persist();
    print("train model")
    model= trainEvaluateModel(trainData, validationData, "entropy", 10, 200)
    print("predict")
    PredictTestData(model)
    print("eval model")    
    AUC = evaluateModel(model, validationData)
    print(AUC)
    


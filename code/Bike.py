from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.mllib.evaluation import BinaryClassificationMetrics, RegressionMetrics

global Path
Path="file:/home/u0416069/hw4/"
'''
What has changed?
1. data/train.csv -> data/hour.csv
'''

def MkDataSource():
    def DataPreprocess(url):
        rawDataWithHeader = sc.textFile(url)
        rawheader = rawDataWithHeader.first()
        rawData = rawDataWithHeader.filter(lambda x:x !=rawheader)
        #rData = ra wData.map(lambda x: x.replace("\"", ""))
        print("rawData = ")
        print(rawData.take(5))
        lines = rawData.map(lambda x: x.split(","))
        print("lines = ")
        print(lines.take(5))
        print('rawHeader = ')
        header = rawheader.split(",")
        print(rawheader)
        print("header = ")
        print(header)
        return lines
    categoriesMap = []
    def predict_features(field,featureEnd):
        c_2 = [float(col) for col in field[2]]
        c_4to13 = [float(col) for col in field[4:featureEnd-2]]
        c_16 = [float(field[16])]
        #print(c_16)
        temp = np.concatenate((c_2, c_4to13, c_16))
        return temp
    def extract_features(field,featureEnd):
        c_2 = [float(col) for col in field[2]]
        c_4to13 = [float(col) for col in field[4:featureEnd-2]]
        temp = np.concatenate((c_2, c_4to13))
        return temp
    def extract_label(record):
        label=(record[-1])
        #print("--label:")
        #print(label)
        return float(label)
    def CreateSparkContext():
        def SetLogger( sc ):
            logger = sc._jvm.org.apache.log4j
            logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
            logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
            logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

        sparkConf = SparkConf().setAppName("RunDecisionTreeRegression").set("spark.ui.showConsoleProgress", "false")
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
        lines = DataPreprocess(Path+"dataset/hour.csv")
        label = lines.first()
        print("----lines----")
        print(lines.take(2))
        (trainLine, validLine) = lines.randomSplit([7, 3])
        trainData = trainLine.map(lambda r:LabeledPoint(extract_label(r),extract_features(r,len(r)-1)))
        validationData = validLine.map(lambda r:LabeledPoint(extract_label(r),extract_features(r,len(r)-1)))
        print(trainData.take(1))
        return (trainData, validationData)
    def PredictTestData(model):
        nonlocal categoriesMap
        nonlocal validLine
        dataRDD = validLine.map(lambda r: ( r[0] ,predict_features(r,len(r)-1 )))
        for data in dataRDD.take(10):
            predictResult = model.predict(data[1])
            #print(data[1])
            pr = 'season: '+ str(data[1][0])
            pr = pr + ' mnth:' + str(data[1][1])
            pr = pr + ' hr: ' + str(data[1][2])
            pr = pr + ' holiday: ' + str(data[1][3])
            pr = pr + ' weekday: ' + str(data[1][4])
            pr = pr + ' workingday: ' + str(data[1][5])
            pr = pr + ' weathersi: ' + str(data[1][6])
            pr = pr + ' temp: ' + str(data[1][7])
            pr = pr + ' atemp: ' + str(data[1][8])
            pr = pr + ' hum: ' + str(data[1][9])
            pr = pr + ' windspeed: ' + str(data[1][10])
            pr = pr + ' cnt: ' + str(data[1][11])
            pr = pr + ' prediction: ' + str(predictResult)
            print(pr)
            #print(data[1])
            #print(predictResult)
            #print(" url：" +str(data[0][:20]) +".... ==>prediction:"+ str(predictResult))

    return (PrepareTrainData,PredictTestData)

def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels=score.zip(validationData.map(lambda p: p.label))
    metrics = RegressionMetrics(scoreAndLabels)
    AUC=metrics.rootMeanSquaredError
    return AUC

def trainEvaluateModel(trainData,validationData,impurityParm, maxDepthParm, maxBinsParm):
    model = DecisionTree.trainRegressor(trainData,
                categoricalFeaturesInfo={},
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
    model= trainEvaluateModel(trainData, validationData, "variance", 10, 100)
    print("predict")
    PredictTestData(model)
    print("eval model")    
    AUC = evaluateModel(model, validationData)
    print(AUC)

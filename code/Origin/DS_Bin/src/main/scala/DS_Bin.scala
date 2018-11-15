import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.storage.StorageLevel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoderEstimator,VectorAssembler}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object RunDecisionTreeBinary {
    case class Data(url: String,
                    urlid: Int,
                    boilerplate: String,
                    alchemyCategory: String,
                    alchemyCategoryScore: Double,
                    avglinksize: Double,
                    commonlinkratio1: Double,
                    commonlinkratio2: Double,
                    commonlinkratio3: Double,
                    commonlinkratio4: Double,
                    compression_ratio: Double,
                    embedRatio: Double,
                    framebased: Double,
                    frameTagRatio: Double,
                    hasDomainLink: Double,
                    htmlRatio: Double,
                    imageRatio: Double,
                    isNews: Double,
                    lengthyLinkDomain: Double,
                    linkwordscore: Double,
                    newsFrontPage: Double,
                    non_markup_alphanum_characters: Double,
                    numberOfLinks: Double,
                    numwordsInUrl: Double,
                    parametrizedLinkRatio: Double,
                    spellingErrorsRatio: Double,
                    label: Double)
    case class Data2(url: String,
                    alchemyCategory: String,
                    alchemyCategoryScore: Double,
                    avglinksize: Double,
                    commonlinkratio1: Double,
                    commonlinkratio2: Double,
                    commonlinkratio3: Double,
                    commonlinkratio4: Double,
                    compression_ratio: Double,
                    embedRatio: Double,
                    framebased: Double,
                    frameTagRatio: Double,
                    hasDomainLink: Double,
                    htmlRatio: Double,
                    imageRatio: Double,
                    isNews: Double,
                    lengthyLinkDomain: Double,
                    linkwordscore: Double,
                    newsFrontPage: Double,
                    non_markup_alphanum_characters: Double,
                    numberOfLinks: Double,
                    numwordsInUrl: Double,
                    parametrizedLinkRatio: Double,
                    spellingErrorsRatio: Double,
                    label: Double)

  def main(args: Array[String]): Unit = {
    SetLogger()
    val spark = SparkSession.builder().appName("Spark SQL basic example").master("local[4]").config("spark.ui.showConsoleProgress","false").getOrCreate()
    import spark.implicits._
    val sch = org.apache.spark.sql.Encoders.product[Data].schema
    println("read data")
    val ds = spark.read.format("csv").option("header", "true").option("delimiter", "\t").schema(sch).load("file:/home/u0416069/data/train.csv").as[Data]
    val xy = ds.randomSplit(Array(0.7,0.3))
    val x = xy(0)
    val y = xy(1)
    val dc = ds.columns.slice(4,27)
    val dc2 = Array("alchemyCategory")
    val dc3 = (dc2 ++ dc)
    val row_ds = ds.select("url",dc3:_*).as[Data2]
    println("setup pipeline")
    val df = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features").setImpurity("gini").setMaxDepth(10).setMaxBins(14)
    val stringIndexer = new StringIndexer().setInputCol("alchemyCategory").setOutputCol("alchemyCategoryIndex")
    val encoder = new OneHotEncoderEstimator().setInputCols(Array("alchemyCategoryIndex")).setOutputCols(Array("alchemyCategoryIndexVec")).setDropLast(false)
    val assemblerInputs = (Array("alchemyCategoryIndexVec") ++ dc).init
    val assembler = new VectorAssembler().setInputCols(assemblerInputs).setOutputCol("features")
    val pipeline = new Pipeline().setStages(Array(stringIndexer,encoder ,assembler,df))
    println("train model")
    val pipelineModel = pipeline.fit(x)
    print("predict")
    val predicted=pipelineModel.transform(y).select("url","prediction").show(10)
    println(predicted)
    println("eval model")
    val evaluator = new BinaryClassificationEvaluator().setRawPredictionCol("rawPrediction").setLabelCol("label").setMetricName("areaUnderROC")
    val predictions =pipelineModel.transform(y)
    val auc= evaluator.evaluate(predictions)
    println(auc)
  }

  def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }
}

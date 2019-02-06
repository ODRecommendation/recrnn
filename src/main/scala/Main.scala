import java.io.{BufferedWriter, File, FileOutputStream, OutputStreamWriter}
import java.nio.file.Paths

import com.amazonaws.auth.DefaultAWSCredentialsProviderChain
import com.amazonaws.regions.Regions
import com.amazonaws.services.s3.AmazonS3ClientBuilder
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.feature.{StringIndexerModel, _}
import org.apache.spark.ml.mleap.SparkUtil
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import resource.managed

import scala.collection.mutable

/**
  * Created by luyangwang on Dec, 2018
  *
  */

case class ModelParams(
                        maxLength: Int,
                        maxEpoch: Int,
                        batchSize: Int,
                        embedOutDim: Int,
                        inputDir: String,
                        logDir: String,
                        dataName: String,
                        lookUpFileName: String,
                        stringIndexerName: String,
                        rnnName: String
                      )

object Main{

  private val s3client = AmazonS3ClientBuilder.standard()
    .withRegion(Regions.CN_NORTH_1)
    .withCredentials(new DefaultAWSCredentialsProviderChain())
    .build()
  val currentDir: String = Paths.get(".").toAbsolutePath + "/"

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)

    System.setProperty("log4j.appender.NotConsole","org.apache.log4j.RollingFileAppender")
    System.setProperty("log4j.appender.NotConsole.fileName","./model.log")
    System.setProperty("log4j.appender.NotConsole.maxFileSize","20MB")

    val params = ModelParams(
      maxLength = 10,
      maxEpoch = 1,
      batchSize = 8,
      embedOutDim = 300,
      inputDir = "./modelFiles/",
      logDir = "./log/",
      dataName = "recRNNSample.csv",
      lookUpFileName = "skuLookUp",
      stringIndexerName = "skuIndexer",
      rnnName = "rnnModel"
    )

    // construct BigDL session
    val conf = new SparkConf()
      .setAppName("recRNN")
      .setMaster("local[*]")
      .set("spark.driver.memory", "100g")
    val sc = NNContext.initNNContext(conf)
    val spark = SparkSession.builder().config(sc.getConf).getOrCreate()

    val (sessionDF, skuCount, skuIndexerModel) = loadPublicData(spark, params)
    val (trainSample, outSize) = assemblyFeature(sessionDF, skuCount, skuIndexerModel, params)

    // train rnn model using Keras API
    val kerasRNN = new SessionRecommender()
//    val model = kerasRNN.buildModel(outSize, skuCount, params.maxLength, params.embedOutDim)
//    kerasRNN.train(model, trainSample, params.inputDir, params.rnnName, params.logDir, params.maxEpoch, params.batchSize)
//    kerasRNN.predict(params.inputDir + params.rnnName + "Keras", trainSample)

    /*Train rnn model using BigDL*/
//    val rnn = new BigDLRNN()
//    val model2 = rnn.buildModel(outSize, skuCount, params.maxLength, params.embedOutDim)
//    rnn.train(model2, trainSample, params.inputDir, params.rnnName, params.logDir, params.maxEpoch, params.batchSize)
  }

  //  Load data using spark session interface
  def loadPublicData(spark: SparkSession, params: ModelParams): (DataFrame, Int, StringIndexerModel) = {
    // stringIndex SKU number
    val sessionDF = spark.read.options(Map("header" -> "true", "delimiter" -> "|")).csv(params.inputDir + params.dataName)
    sessionDF.printSchema()

    val skuCount = sessionDF.select("SKU_NUM").distinct().count().toInt
    println(skuCount)
    val skuIndexer = new StringIndexer().setInputCol("SKU_NUM").setOutputCol("SKU_INDEX").setHandleInvalid("keep")
    val skuIndexerModel = skuIndexer.fit(sessionDF)
    skuIndexerModel.write.overwrite().save(params.inputDir + params.stringIndexerName)
    saveToMleap(skuIndexerModel, sessionDF, params.stringIndexerName)
    println("SkuIndexerModel has been saved")
    (sessionDF, skuCount, skuIndexerModel)
  }

  // convert features to RDD[Sample[FLoat]]
  def assemblyFeature(sessionDF: DataFrame, skuCount: Int, skuIndexerModel: StringIndexerModel, params: ModelParams) = {
    // stringIndex the sku number and adjust the starting index to 1
    val indexedDF = skuIndexerModel
      .transform(sessionDF)
      .withColumn("SKU_INDEX", col("SKU_INDEX") + 1)

    // save lookUp table for index to string revert
    val lookUp = indexedDF.select("SKU_NUM", "SKU_INDEX").distinct()
      .rdd.map(x => {
      val text = x.getString(0)
      val label = x.getAs[Double](1).toInt
      (text, label)
    }).collect()

    val file = params.inputDir + params.lookUpFileName
    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file)))
    for (x <- lookUp) {
      writer.write(x._1 + " " + x._2 + "\n")
    }
    writer.close()

    indexedDF.show()
    indexedDF.printSchema()

    // collect item to sequence
    val seqDF = indexedDF.groupBy("SESSION_ID")
      .agg(collect_list("SKU_INDEX").alias("sku"))
      .filter(col("sku").isNotNull)

    seqDF.printSchema()
    seqDF.show()

    // prePad UDF
    def prePadding: mutable.WrappedArray[java.lang.Double] => Array[Float] = x => {
      x.array.map(_.toFloat).dropRight(1).reverse.padTo(params.maxLength, 0f).reverse
    }
    val prePaddingUDF = udf(prePadding)

    // get label UDF
    def getLabel: mutable.WrappedArray[java.lang.Double] => Float = x => {
      x.takeRight(1).head.floatValue() + 1
    }
    val getLabelUDF = udf(getLabel)

    val trainDF = seqDF
      .withColumn("features", prePaddingUDF(col("sku")))
      .withColumn("label", getLabelUDF(col("sku")))

    trainDF.show(false)
    trainDF.printSchema()


    val outSize = trainDF.rdd.map(_.getAs[Float]("label")).max.toInt
    println(outSize)

    // dataFrame to sample
    val trainSample = trainDF.rdd.map(r => {
      val label = Tensor[Float](T(r.getAs[Float]("label")))
      val array = r.getAs[mutable.WrappedArray[java.lang.Float]]("features").array.map(_.toFloat)
      val vec = Tensor(array, Array(params.maxLength))
      Sample(vec, label)
    })

    println("Sample feature print: \n"+ trainSample.take(1).head.feature())
    println("Sample label print: \n" + trainSample.take(1).head.label())

    (trainSample, outSize)
  }

  // save mLeap serialized transformer
  def saveToMleap(
                   indexerModel: StringIndexerModel,
                   data: DataFrame,
                   indexerModelPath: String
                 ): Unit = {
    val pipeline = SparkUtil.createPipelineModel(uid = "pipeline", Array(indexerModel))
    val sbc = SparkBundleContext().withDataset(pipeline.transform(data))
    new File(s"$currentDir$indexerModelPath.zip").delete()
    for(bf <- managed(BundleFile(s"jar:file:$currentDir$indexerModelPath.zip"))) {
      pipeline.writeBundle.save(bf)(sbc).get
    }
  }

  def putS3Obj(
                bucketName: String,
                fileKey: String,
                filePath: String
              ): Unit = {
    val file = new File(filePath)
    s3client.putObject(bucketName, fileKey, file)
    println(s"$filePath has been uploaded to s3 at $bucketName$fileKey")
  }

}

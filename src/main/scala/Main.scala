import java.io.{BufferedWriter, File, FileOutputStream, OutputStreamWriter}
import java.nio.file.Paths

import com.amazonaws.auth.DefaultAWSCredentialsProviderChain
import com.amazonaws.regions.Regions
import com.amazonaws.services.s3.AmazonS3ClientBuilder
import com.intel.analytics.bigdl.dataset.{Sample, TensorSample}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.recommendation.{Utils, WideAndDeep}
import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.feature.{StringIndexerModel, _}
import org.apache.spark.ml.mleap.SparkUtil
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, SparkSession}
import resource.managed

import scala.collection.mutable
import scala.util.Random

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
                        rnnData: String,
                        ncfData: String,
                        lookUpFileName: String,
                        userIndexerName: String,
                        itemIndexerName: String,
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
      maxEpoch = 2,
      batchSize = 64,
      embedOutDim = 300,
      inputDir = "./modelFiles/",
      logDir = "./log/",
      rnnData = "rnnData.csv",
      ncfData = "ncfData.csv",
      lookUpFileName = "skuLookUp",
      userIndexerName = "userIndexer",
      itemIndexerName = "itemIndexer",
      rnnName = "rnnModel"
    )

    // construct BigDL session
    val conf = new SparkConf()
      .setAppName("recRNN")
      .setMaster("local[*]")
      .set("spark.driver.memory", "100g")
    val sc = NNContext.initNNContext(conf)
    val spark = SparkSession.builder().config(sc.getConf).getOrCreate()

    val (sessionDF, historyDF, userCount, itemCount, userIndexerModel,itemIndexerModel) = loadPublicData(spark, params)
    val (trainSample, outSize) = assemblyFeature(sessionDF, historyDF, userCount, itemCount, userIndexerModel, itemIndexerModel, params)
    val sr = SessionRecommender[Float](
      userCount, itemCount, outSize + 2
    )
    SessionRecommender.train(
      sr, trainSample, "modelFiles", "sr", "log", params.maxEpoch, params.batchSize
    )

//
//    // train rnn model using Keras API
//    val kerasRNN = new SessionRecommender()
//    val model = kerasRNN.buildModel(outSize, skuCount, params.maxLength, params.embedOutDim)
//    kerasRNN.train(model, trainSample, params.inputDir, params.rnnName, params.logDir, params.maxEpoch, params.batchSize)
//    kerasRNN.predict(params.inputDir + params.rnnName + "Keras", trainSample)

    /*Train rnn model using BigDL*/
//    val rnn = new BigDLRNN()
//    val model2 = rnn.buildModel(outSize, skuCount, params.maxLength, params.embedOutDim)
//    rnn.train(model2, trainSample, params.inputDir, params.rnnName, params.logDir, params.maxEpoch, params.batchSize)
  }

  //  Load data using spark session interface
  def loadPublicData(spark: SparkSession, params: ModelParams) = {
    // stringIndex SKU number
    val sessionDF = spark.read.options(Map("header" -> "true", "delimiter" -> ",")).csv(params.inputDir + params.rnnData)
    sessionDF.printSchema()
    val historyDF = spark.read.options(Map("header" -> "true", "delimiter" -> ",")).csv(params.inputDir + params.ncfData)
    historyDF.printSchema()

    val userCount = historyDF.select("AGENT_ID").distinct().count().toInt + 1
    println("userCount = " + userCount)
    val itemCount = sessionDF.select("SKU_NUM").distinct().count().toInt + 1
    println("itemCount = " + itemCount)
    val userIndexer = new StringIndexer().setInputCol("AGENT_ID").setOutputCol("userId").setHandleInvalid("keep")
    val itemIndexer = new StringIndexer().setInputCol("SKU_NUM").setOutputCol("SKU_INDEX").setHandleInvalid("keep")
    val userIndexerModel = userIndexer.fit(historyDF)
    val itemIndexerModel = itemIndexer.fit(sessionDF)
    userIndexerModel.write.overwrite().save(params.inputDir + params.userIndexerName)
    itemIndexerModel.write.overwrite().save(params.inputDir + params.itemIndexerName)
    saveToMleap(itemIndexerModel, sessionDF, params.userIndexerName)
    saveToMleap(itemIndexerModel, sessionDF, params.itemIndexerName)
    println("SkuIndexerModel has been saved")
    (sessionDF, historyDF, userCount, itemCount, userIndexerModel, itemIndexerModel)
  }

  // convert features to RDD[Sample[FLoat]]
  def assemblyFeature(
                       sessionDF: DataFrame,
                       historyDF: DataFrame,
                       userCount: Int,
                       itemCount: Int,
                       userIndexerModel: StringIndexerModel,
                       itemIndexerModel: StringIndexerModel,
                       params: ModelParams
                     ) = {
//    val joined = sessionDF.join(historyDF, Array("AGENT_ID")).distinct()
//    joined.show()
    // stringIndex the sku number and adjust the starting index to 1
    val indexedSessionDF = itemIndexerModel
      .transform(sessionDF)
      .withColumn("SKU_INDEX", col("SKU_INDEX") + 2)
    val indexedHistoryDF = itemIndexerModel.transform(historyDF)
      .withColumn("SKU_INDEX", col("SKU_INDEX") + 2)
      .select("AGENT_ID", "SKU_INDEX")

    // save lookUp table for index to string revert
    val lookUp = indexedSessionDF.select("SKU_NUM", "SKU_INDEX").distinct()
      .rdd.map(x => {
      val text = x.getString(0)
      val label = x.getAs[Double](1)
      (text, label)
    }).collect()

    val file = params.inputDir + params.lookUpFileName
    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file)))
    for (x <- lookUp) {
      writer.write(x._1 + " " + x._2 + "\n")
    }
    writer.close()

    indexedSessionDF.show()
    indexedSessionDF.printSchema()

    // collect item to sequence
    val seqDF = indexedSessionDF.groupBy("SESSION_ID", "AGENT_ID")
      .agg(collect_list("SKU_INDEX").alias("sku"))
      .filter(col("sku").isNotNull)

    seqDF.printSchema()
    seqDF.show()

    // prePad UDF
    def prePadding: mutable.WrappedArray[java.lang.Double] => Array[Float] = x => {
      x.array.map(_.toFloat).dropRight(1).reverse.padTo(params.maxLength, 1f).reverse
    }
    val prePaddingUDF = udf(prePadding)

    // get label UDF
    def getLabel: mutable.WrappedArray[java.lang.Double] => Float = x => {
      x.takeRight(1).head.floatValue()
    }
    val getLabelUDF = udf(getLabel)

    val rnnDF = seqDF
      .withColumn("rnnItem", prePaddingUDF(col("sku")))
      .withColumn("label", getLabelUDF(col("sku")))
      .select("AGENT_ID", "rnnItem", "label")

    rnnDF.show(false)
    rnnDF.printSchema()

    val outSize = rnnDF.rdd.map(_.getAs[Float]("label")).max.toInt
    println(outSize)

    val seqDF1 = indexedHistoryDF.groupBy("AGENT_ID")
      .agg(collect_list("SKU_INDEX").alias("history"))
      .filter(col("history").isNotNull)

    val cfDF = seqDF1.withColumn("history", prePaddingUDF(col("history")))
    cfDF.show()

    val joined = cfDF.join(rnnDF, Array("AGENT_ID"))
        .filter(size(col("history")) <= 10).filter(size(col("rnnItem")) <= 10)
        .distinct()
    joined.show(false)

//    def getNegative(indexed: DataFrame): DataFrame = {
//      val schema = indexed.schema
//      require(schema.fieldNames.contains("userId"), s"Column userId should exist")
//      require(schema.fieldNames.contains("itemId"), s"Column itemId should exist")
//      require(schema.fieldNames.contains("RATING"), s"Column label should exist")
//
//      val indexedDF = indexed.select("userId", "itemId", "RATING")
//      val minMaxRow = indexedDF.agg(max("userId"), max("itemId")).collect()(0)
//      val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))
//      val sampleDict = indexedDF.rdd.map(row => row(0) + "," + row(1)).collect().toSet
//
//      val dfCount = indexedDF.count.toInt
//
//      import indexed.sqlContext.implicits._
//
//      @transient lazy val ran = new Random(System.nanoTime())
//
//      val negative = indexedDF.rdd
//        .map(x => {
//          val uid = x.getAs[Int](0)
//          val iid = Math.max(ran.nextInt(itemCount), 1)
//          (uid, iid)
//        })
//        .filter(x => !sampleDict.contains(x._1 + "," + x._2)).distinct()
//        .map(x => (x._1, x._2, 1))
//        .toDF("userId", "itemId", "RATING")
//
//      negative
//    }
//    val negative = getNegative(indexedHistoryDF1)
//    val unioned = negative.union(indexedHistoryDF1)
//
//    val joined = rnnDF.join(unioned, Array("userId")).distinct()
//
//    joined.show()

    // dataFrame to sample
    val trainSample = joined.rdd.map(r => {
      val label = Tensor[Float](T(r.getAs[Float]("label")))
      val mlpFeature = r.getAs[mutable.WrappedArray[java.lang.Float]]("history").array.map(_.toFloat)
      val rnnFeature = r.getAs[mutable.WrappedArray[java.lang.Float]]("rnnItem").array.map(_.toFloat)
      val mlpSample = Tensor(mlpFeature, Array(params.maxLength))
      val rnnSample = Tensor(rnnFeature, Array(params.maxLength))
      TensorSample[Float](Array(mlpSample, rnnSample), Array(label))
//      val feature = Array(mlpFeature, rnnFeature)
//      Sample(feature, label)
    })

    println("Sample feature print: \n"+ trainSample.take(1).head.feature(0))
    println("Sample feature print: \n"+ trainSample.take(1).head.feature(1))
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

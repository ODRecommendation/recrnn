import java.io.File
import java.nio.file.Paths

import com.amazonaws.auth.DefaultAWSCredentialsProviderChain
import com.amazonaws.regions.Regions
import com.amazonaws.services.s3.AmazonS3ClientBuilder
import com.amazonaws.services.s3.model.PutObjectResult
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

    val params = ModelParams(
      maxLength = 10,
      maxEpoch = 100,
      batchSize = 2560,
      embedOutDim = 200,
      inputDir = "./modelFiles/",
      logDir = "./log/",
      dataName = "recrnn.csv",
      stringIndexerName = "skuIndexer",
      rnnName = "rnnModel"
    )

    /*Construct BigDL session*/
    val conf = new SparkConf()
      .setAppName("recRNN")
      .setMaster("local[*]")
      .set("spark.driver.memory", "100g")
    val sc = NNContext.initNNContext(conf)
    val spark = SparkSession.builder().config(sc.getConf).getOrCreate()

    /*StringIndex SKU number*/
    val data = spark.read.options(Map("header" -> "true", "delimiter" -> "|")).csv(params.inputDir + params.dataName)

    val skuCount = data.select("SKU_NUM").distinct().count().toInt
    println(skuCount)
    val skuIndexer = new StringIndexer().setInputCol("SKU_NUM").setOutputCol("SKU_INDEX").setHandleInvalid("keep")
    val skuIndexerModel = skuIndexer.fit(data)
    skuIndexerModel.write.overwrite().save(params.inputDir + params.stringIndexerName)
    saveToMleap(skuIndexerModel, data, params.stringIndexerName)
    println("SkuIndexerModel has been saved")

    /*StringIndex the sku number and adjust the starting index to 1*/
    val data1 = skuIndexerModel
      .transform(data)
      .withColumn("SKU_INDEX", col("SKU_INDEX") + 1)

    data1.show()
    data1.printSchema()

    /*Collect item to sequence*/
    val data2 = data1.groupBy("SESSION_ID")
      .agg(collect_list("SKU_INDEX").alias("sku"))
      .filter(col("sku").isNotNull)

    data2.printSchema()
    data2.show()

    /*PrePad UDF*/
    def prePadding: mutable.WrappedArray[java.lang.Double] => Array[Float] = x => {
      val item = if (x.length > params.maxLength) x.array.map(_.toFloat)
      else Array.fill[Float](params.maxLength - x.length + 1)(0) ++ x.array.map(_.toFloat)
      val item2 = item.takeRight(params.maxLength + 1)
      val item3 = item2.dropRight(1)
      item3
    }
    val prePaddingUDF = udf(prePadding)

    /*Get label UDF*/
    def getLabel: mutable.WrappedArray[java.lang.Double] => Float = x => {
      x.takeRight(1).head.floatValue() + 1
    }
    val getLabelUDF = udf(getLabel)

    val data3 = data2
      .withColumn("features", prePaddingUDF(col("sku")))
      .withColumn("label", getLabelUDF(col("sku")))

    data3.show()
    data3.printSchema()

    val outSize = data3.rdd.map(_.getAs[Float]("label")).max.toInt
    println(outSize)

    /*DataFrame to sample*/
    val trainSample = data3.rdd.map(r => {
      val label = Tensor[Float](T(r.getAs[Float]("label")))
      val array = r.getAs[mutable.WrappedArray[java.lang.Float]]("features").array.map(_.toFloat)
      val vec = Tensor(array, Array(params.maxLength))
      Sample(vec, label)
    })

    println("Sample feature print: \n"+ trainSample.take(1).head.feature())
    println("Sample label print: \n" + trainSample.take(1).head.label())

    /*Train rnn model using Keras API*/
    val kerasRNN = new KerasRNN()
    val model1 = kerasRNN.buildModel(outSize, skuCount, params.maxLength, params.embedOutDim)
    kerasRNN.train(model1, trainSample, params.inputDir, params.rnnName, params.logDir, params.maxEpoch, params.batchSize)

    /*Train rnn model using BigDL*/
//    val rnn = new BigDLRNN()
//    val model2 = rnn.buildModel(outSize, skuCount, params.maxLength, params.embedOutDim)
//    rnn.train(model2, trainSample, params.inputDir, params.rnnName, params.logDir, params.maxEpoch, params.batchSize)
  }

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

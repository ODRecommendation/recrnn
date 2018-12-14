import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.example.utils.TextClassifier
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

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
                 logDir: String
                 )

object Main{

  def main(args: Array[String]): Unit = {

    /*Construct BigDL session*/
    Logger.getLogger("org").setLevel(Level.WARN)

    val params = ModelParams(
      maxLength = 5,
      maxEpoch = 20,
      batchSize = 56,
      embedOutDim = 50,
      inputDir = "s3://philipsbigdata/testRNN/",
      logDir = "./log/"
    )
    val conf = new SparkConf()
      .setAppName("recRNN")
//      .setMaster("local[*]")
//      .set("spark.driver.memory", "100g")
    val sc = NNContext.initNNContext(conf)
    val spark = SparkSession.builder().config(conf).getOrCreate()

    /*One hot encode each item*/
    val data = spark.read.options(Map("header" -> "true", "delimiter" -> "|")).csv(params.inputDir + "recrnn.csv")

    val skuCount = data.select("SKU_NUM").distinct().count().toInt
    println(skuCount)
    val skuIndexer = new StringIndexer().setInputCol("SKU_NUM").setOutputCol("SKU_INDEX").setHandleInvalid("keep")
    val skuIndexerModel = skuIndexer.fit(data)
    val ohe = new OneHotEncoder().setInputCol("SKU_INDEX").setOutputCol("vectors")

    val data1a = skuIndexerModel.transform(data).withColumn("SKU_INDEX", col("SKU_INDEX") + 1)
    val data1 = ohe.transform(data1a)

    data1.show()
    data1.printSchema()

    /*Collect item to sequence*/
    val data2 = data1.groupBy("SESSION_ID")
      .agg(collect_list("vectors").alias("item"), collect_list("SKU_INDEX").alias("sku"))
      .filter(col("sku").isNotNull && col("item").isNotNull)

    data2.printSchema()
    data2.show()

    val skuPadding = Array.fill[Float](skuCount)(0)
    val skuPadding1 = Array.fill[Array[Float]](params.maxLength)(skuPadding)
    val bcPadding = sc.broadcast(skuPadding1).value

    /*Pad items to equal length*/
    def prePadding: mutable.WrappedArray[SparseVector] => Array[Array[Float]] = x => {
      val skuPadding = Array.fill[Float](skuCount)(0)
      val skuPadding1 = Array.fill[Array[Float]](params.maxLength)(skuPadding)
      val item = skuPadding1 ++ x.array.map(_.toArray.map(_.toFloat))
      val item2 = item.takeRight(params.maxLength + 1)
      val item3 = item2.dropRight(1)
      item3
    }

    def getLabel: mutable.WrappedArray[java.lang.Double] => Float = x => {
      x.takeRight(1).head.floatValue()
    }

    val prePaddingUDF = udf(prePadding)
    val getLabelUDF = udf(getLabel)

    val data3 = data2
      .withColumn("features", prePaddingUDF(col("item")))
      .withColumn("label", getLabelUDF(col("sku")))

    data3.show()
    data3.printSchema()


    val outSize = data3.rdd.map(_.getAs[Float]("label")).max.toInt
    println(outSize)

    /*Dataframe to tensor*/
    val trainSample = data3.rdd.map(r => {
      val label = Tensor[Float](T(r.getAs[Float]("label")))
      val array = r.getAs[mutable.WrappedArray[mutable.WrappedArray[Float]]]("features").array.flatten
      val vec = Tensor(array, Array(params.maxLength, skuCount))
      Sample(vec, label)
    })

    println("Sample feature print: "+ trainSample.take(1).head.feature())
    println("Sample label print: " + trainSample.take(1).head.label())

    /*Train rnn model*/
    val rnn = new RecRNN()
    val model = rnn.buildModel(outSize, skuCount, params.maxLength, params.embedOutDim)
    rnn.train(model, trainSample, params.inputDir, params.logDir, params.maxEpoch, params.batchSize)
  }
}

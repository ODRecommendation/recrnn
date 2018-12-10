import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

import scala.collection.mutable

/**
  * Created by luyangwang on Dec, 2018
  *
  */
object Main extends App{

  Logger.getLogger("org").setLevel(Level.WARN)

  val conf = new SparkConf()
    .setAppName("stemCell")
    .setMaster("local[*]")
    .set("spark.driver.memory", "100g")
  val sc = NNContext.initNNContext(conf)
  val spark = SparkSession.builder().config(conf).getOrCreate()

  val data = spark.read.options(Map("header" -> "true", "delimiter" -> "|")).csv("./modelFiles/recRNNsample.csv")

  val skuCount = data.select("SKU_NUM").distinct().count().toInt
  val skuIndexer = new StringIndexer().setInputCol("SKU_NUM").setOutputCol("SKU_INDEX").setHandleInvalid("keep")
  val labelIndexer = new StringIndexer().setInputCol("out").setOutputCol("label").setHandleInvalid("keep")

  val skuIndexerModel = skuIndexer.fit(data)

  val data1 = skuIndexerModel.transform(data)
    .withColumn("SKU_INDEX", col("SKU_INDEX")+1)

  data1.show()
  data1.printSchema()

  val data2 = data1.groupBy("SESSION_ID")
    .agg(collect_list("SKU_INDEX").alias("item"), collect_list("SKU_NUM").alias("SKU"))

  data2.show()
  data2.printSchema()

  val skuPadding = Array.fill[Double](10)(0.0)
  val bcPadding = sc.broadcast(skuPadding).value

  import spark.implicits._
  val data3 = data2.rdd.map(r => {
    val item = r.getAs[mutable.WrappedArray[java.lang.Double]]("item").array.map(_.doubleValue())
    val labelRaw = r.getAs[mutable.WrappedArray[String]]("SKU").array
    val item1 = bcPadding ++ item
    val item2 = item1.takeRight(11)
    val label = labelRaw.takeRight(1).head
    val features = item2.dropRight(1)
    (features, label)
  }).toDF("features", "out")

  val labelIndexerModel = labelIndexer.fit(data3)

  val data4 = labelIndexerModel.transform(data3).rdd.map(r => {
    val features = r.getAs[mutable.WrappedArray[java.lang.Double]]("features").array.map(_.doubleValue())
    val label = r.getAs[Double]("label") + 1.0
    (features, label)
  })

  val outSize = data4.map(_._2).distinct().count().toInt
  println(outSize)

  val trainSample = data4.map(r => {
    val label = Tensor[Double](T(r._2))
    val array = r._1
    val vec = Tensor(Storage(array), 1, Array(10))
    Sample(vec, label)
  })

  println("Sample feature print: "+ trainSample.take(1).head.feature())
  println("Sample label print: " + trainSample.take(1).head.label())

  val rnn = new RecRNN()
  val model = rnn.buildModel(outSize, skuCount)
  rnn.train(model, trainSample, "./modelFiles/rnnModel", 100, 8)

}

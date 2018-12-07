import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import model.RecRNN
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
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
//      .groupBy("SESSION_ID").agg(collect_list("SKU_NUM").alias("SKU"))
//  data.show()

  val data1 = data.groupBy("SESSION_ID").agg(collect_list("SKU_NUM").alias("SKU"))

  val skuIndexer = new StringIndexer().setInputCol("SKU_NUM").setOutputCol("SKU_INDEX").setHandleInvalid("keep")
  val labelIndexer = new StringIndexer().setInputCol("out").setOutputCol("label").setHandleInvalid("keep")
  val oheIndexer = new OneHotEncoder().setInputCol("SKU_INDEX").setOutputCol("item")
  val pipeline = new Pipeline().setStages(Array(skuIndexer, oheIndexer))
//  val pipelineModel = pipeline.fit(data)

  val skuPadding = Array.fill[Double](200)(0)
  val skuPadding1 = Array.fill[Array[Double]](5)(skuPadding)

  val word2Vec = new Word2Vec().setInputCol("SKU").setOutputCol("Vec").setVectorSize(200).setMinCount(0)
  val vecModel = word2Vec.fit(data1)
  vecModel.write.overwrite().save("./modelFiles/skuEmbedding")
  val result = vecModel.transform(data1)
  result.show()

  val lookUp = vecModel.getVectors.withColumnRenamed("word", "SKU_NUM")
  lookUp.show()
  lookUp.printSchema()

  val data2 = data.join(lookUp, Seq("SKU_NUM"))
  data2.show()

  import spark.implicits._
  val data3 = data2
    .groupBy("SESSION_ID").agg(collect_list("vector").alias("item"), collect_list("SKU_NUM").alias("sku"))
      .rdd.map(r => {
//    val session = r.getAs[String]("SESSION_ID")
    val item = r.getAs[mutable.WrappedArray[DenseVector]]("item").array.map(x=>x.toArray)
    val item1 = skuPadding1 ++ item
    val item2 = item1.takeRight(6)
    val label = r.getAs[mutable.WrappedArray[String]]("sku").array.takeRight(1).head
    val features = item2.dropRight(1)
    (features, label)
  })
    .toDF("SKU", "out")

  val labelIndexerModel = labelIndexer.fit(data3)
  val data4 = labelIndexerModel.transform(data3)
    .withColumn("label", col("label") + 1)

  val outSize = data4.select("label").distinct().count().toInt

  val trainSample = data4.rdd.map(r => {
    val label = Tensor[Double](T(r.getAs[Double]("label")))
    val array = r.getAs[mutable.WrappedArray[mutable.WrappedArray[Double]]]("SKU").array.map(x=>x.toArray)
    val vec = Tensor(array.flatten, Array(5, 200))
    Sample(vec, label)
  })

  println("Sample feature print: "+ trainSample.take(1).head.feature())
  println("Sample label print: " + trainSample.take(1).head.label())

  val rnn = new RecRNN()
  val model = rnn.buildModel(outSize)
  rnn.train(model, trainSample, "./modelFiles/rnnModel", 4)

}

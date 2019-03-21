import java.io.Serializable

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable
import scala.util.Random

/**
  * Created by luyangwang on Mar, 2019
  *
  */


class SessionRecSpec extends FlatSpec with Matchers with BeforeAndAfter {

  "SessionRecommender without history" should "work properly" in {
    val itemCount = 100
    val embedOutDim = 300
    val maxLength = 10
    val input = Tensor(Array.fill[Float](maxLength)((Random.nextInt(99) + 1).toFloat), Array(1, maxLength))

    val sr = SessionRecommender[Float](
      itemCount = itemCount,
      numClasses = itemCount + 1,
      itemEmbed = embedOutDim,
      includeHistory = false,
      maxLength = maxLength
    )
    val output = sr.forward(input)
    val gradInput = sr.backward(input, output)
  }

  "SessionRecommender with history" should "work properly" in {
    val itemCount = 100
    val embedOutDim = 300
    val maxLength = 10
    val input = T(
      Array.fill[Float](maxLength)((Random.nextInt(99) + 1).toFloat), Array(1, maxLength),
      Array.fill[Float](maxLength)((Random.nextInt(99) + 1).toFloat), Array(1, maxLength)
    )

    val sr = SessionRecommender[Float](
      itemCount = itemCount,
      numClasses = itemCount + 1,
      itemEmbed = embedOutDim,
      includeHistory = true,
      maxLength = maxLength
    )
    val output = sr.forward(input)
    val gradInput = sr.backward(input, output)
  }

}

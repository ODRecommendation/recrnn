import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
//import com.intel.analytics.bigdl.nn.keras._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table
import org.apache.spark.rdd.RDD
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


//val seed = 100
////
//val inputFrameSize = 5
//val outputFrameSize = 3
//val kW = 5
//val dW = 3
//val layer = TemporalConvolution(inputFrameSize, outputFrameSize,kW)
//
//val input = Tensor(10,5)
//println(input)
////val gradOutput = Tensor(3, 3).apply1(e => Random.nextFloat())
//
//val output = layer.updateOutput(input)
//println(output)

//val layer = LookupTable(9, 4, 2, 0.1, 2.0, true)
//val input = Tensor(Storage(Array(5.0f, 2.0f, 6.0f, 9.0f, 4.0f)), 1, Array(5))
//
//val output = layer.forward(input)
//val gradInput = layer.backward(input, output)

import com.intel.analytics.bigdl.nn.JoinTable
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

val layer = JoinTable(2, 2)
val input1 = Tensor(T(
  T(
    T(1f, 2f, 3f),
    T(2f, 3f, 4f),
    T(3f, 4f, 5f))
))

val input2 = Tensor(T(
  T(
    T(3f, 4f, 5f),
    T(2f, 3f, 4f),
    T(1f, 2f, 3f))
))

val input = T(input1, input2)

val gradOutput = Tensor(T(
  T(
    T(1f, 2f, 3f, 3f, 4f, 5f),
    T(2f, 3f, 4f, 2f, 3f, 4f),
    T(3f, 4f, 5f, 1f, 2f, 3f)
  )))

val output = layer.forward(input)
val grad = layer.backward(input, gradOutput)

println(output)
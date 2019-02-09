import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.models.recommendation.Recommender

import scala.reflect.ClassTag

/**
  * Created by luyangwang on Dec, 2018
  *
  */

/**
  * The factory method to create a SessionRecommender instance.
  *
  * @param itemCount The number of distinct items. Positive integer.
  * @param numClasses The number of classes. Positive integer.
  * @param itemEmbed The output size of embedding layer. Positive integer.
  * @param mlpHiddenLayers Units of hidden layers for the mlp model. Array of positive integers. Default is Array(40, 20, 10).
  * @param includeCF Whether to include purchase history. Boolean. Default is true.
  * @param maxLength The max number of tokens
  */
class SessionRecommender[T: ClassTag](
                             val itemCount: Int,
                             val numClasses: Int,
                             val itemEmbed: Int = 300,
                             val mlpHiddenLayers: Array[Int] = Array(40, 20, 10),
                             val includeCF: Boolean = true,
                             val maxLength: Int = 10
                                     )(implicit ev: TensorNumeric[T])
  extends Recommender[T] {

  override def buildModel(): AbstractModule[Tensor[T], Tensor[T], T] = {

    val fullModel = Sequential[T]()

    val mlp = Sequential[T]().setName("mlp")

    val mlpItemTable = LookupTable[T](itemCount, itemEmbed)
    mlpItemTable.setWeightsBias(Array(Tensor[T](itemCount, itemEmbed).randn(0, 0.1)))
    val mlpEmbeddedLayer = Sequential[T]().add(mlpItemTable)
    mlp.add(mlpEmbeddedLayer).add(Select(2, -1))
    val linear1 = Linear[T](itemEmbed, mlpHiddenLayers(0))
    mlp.add(linear1).add(ReLU())
    for (i <- 1 until mlpHiddenLayers.length) {
      mlp.add(Linear(mlpHiddenLayers(i - 1), mlpHiddenLayers(i))).add(ReLU())
    }
    mlp.add(Linear(mlpHiddenLayers.last, numClasses))

    val rnn = Sequential[T]().setName("rnn")

    val rnnTable = LookupTable[T](itemCount, itemEmbed)
    rnnTable.setWeightsBias(Array(Tensor[T](itemCount, itemEmbed).randn(0, 0.1)))
    val rnnEmbeddedLayer = Sequential[T]().add(rnnTable)
    rnn.add(rnnEmbeddedLayer)
      .add(Recurrent[T]().add(GRU(itemEmbed, 200)))
      .add(Select(2, -1))
      .add(Linear(200, numClasses))

    fullModel
      .add(ParallelTable().add(mlp).add(rnn))
      .add(CAveTable()).add(LogSoftMax())

    fullModel.add(LogSoftMax())
    fullModel.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object SessionRecommender {
  /**
    * The factory method to create a SessionRecommender instance.
    *
    * @param itemCount The number of distinct items. Positive integer.
    * @param numClasses The number of classes. Positive integer.
    * @param itemEmbed The output size of embedding layer. Positive integer.
    * @param mlpHiddenLayers Units of hidden layers for the deep model. Array of positive integers.
    *                     Default is Array(40, 20, 10).
    * @param includeCF Whether to include purchase history. Boolean. Default is true.
    * @param maxLength The max number of tokens
    */
  def apply[@specialized(Float, Double) T: ClassTag](
      itemCount: Int,
      numClasses: Int,
      itemEmbed: Int = 20,
      mlpHiddenLayers: Array[Int] = Array(40, 20, 10),
      includeCF: Boolean = true,
      maxLength: Int = 10)
      (implicit ev: TensorNumeric[T]): SessionRecommender[T] = {
    new SessionRecommender[T](
      itemCount,
      numClasses,
      itemEmbed,
      mlpHiddenLayers,
      includeCF,
      maxLength
    ).build()
  }

  /**
    * Load an existing WideAndDeep model (with weights).
    *
    * @param path The path for the pre-defined model.
    *             Local file system, HDFS and Amazon S3 are supported.
    *             HDFS path should be like "hdfs://[host]:[port]/xxx".
    *             Amazon S3 path should be like "s3a://bucket/xxx".
    * @param weightPath The path for pre-trained weights if any. Default is null.
    * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
    */
  def loadModel[T: ClassTag](
                              path: String,
                              weightPath: String = null
                            )(implicit ev: TensorNumeric[T]): SessionRecommender[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[SessionRecommender[T]]
  }

}

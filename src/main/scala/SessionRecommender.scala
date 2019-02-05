import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.models.recommendation.Recommender
//import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
  * Created by luyangwang on Dec, 2018
  *
  */
class SessionRecommender[T: ClassTag](
                             val userCount: Int,
                             val itemCount: Int,
                             val numClasses: Int,
                             val userEmbed: Int = 20,
                             val itemEmbed: Int = 20,
                             val hiddenLayers: Array[Int] = Array(40, 20, 10),
                             val includeMF: Boolean = true,
                             val mfEmbed: Int = 20,
                             val maxLength: Int = 10
                                     )(implicit ev: TensorNumeric[T])
  extends Recommender[T] {

  override def buildModel(): AbstractModule[Tensor[T], Tensor[T], T] = {
    val model = Sequential[T]()
    val ncf = Sequential[T]()
    val rnn = Sequential[T]()

    val mlpUserTable = LookupTable[T](userCount, userEmbed)
    val mlpItemTable = LookupTable[T](itemCount, itemEmbed)
    val rnnTable = LookupTable[T](itemCount, itemEmbed)
    mlpUserTable.setWeightsBias(Array(Tensor[T](userCount, userEmbed).randn(0, 0.1)))
    mlpItemTable.setWeightsBias(Array(Tensor[T](itemCount, itemEmbed).randn(0, 0.1)))
    rnnTable.setWeightsBias(Array(Tensor[T](itemCount, itemEmbed).randn(0, 0.1)))
    val mlpEmbeddedLayer = Concat[T](2)
      .add(Sequential[T]().add(Select(2, 1)).add(mlpUserTable))
      .add(Sequential[T]().add(Select(2, 2)).add(mlpItemTable))
    val mlpModel = Sequential[T]()
    mlpModel.add(mlpEmbeddedLayer)
    val linear1 = Linear[T](itemEmbed + userEmbed, hiddenLayers(0))
    mlpModel.add(linear1).add(ReLU())
    for (i <- 1 until hiddenLayers.length) {
      mlpModel.add(Linear(hiddenLayers(i - 1), hiddenLayers(i))).add(ReLU())
    }

    if (includeMF) {
      require(mfEmbed > 0, s"please provide meaningful number of embedding units")
      val mfUserTable: LookupTable[T] = LookupTable[T](userCount, mfEmbed)
      val mfItemTable = LookupTable[T](itemCount, mfEmbed)
      mfUserTable.setWeightsBias(Array(Tensor[T](userCount, mfEmbed).randn(0, 0.1)))
      mfItemTable.setWeightsBias(Array(Tensor[T](itemCount, mfEmbed).randn(0, 0.1)))
      val mfEmbeddedLayer = ConcatTable()
        .add(Sequential[T]().add(Select(2, 1)).add(mfUserTable))
        .add(Sequential[T]().add(Select(2, 2)).add(mfItemTable))
      val mfModel = Sequential[T]()
      mfModel.add(mfEmbeddedLayer).add(CMulTable())
      val concatedModel = Concat(2).add(mfModel).add(mlpModel)
      ncf.add(concatedModel)
    }
    else {
      ncf.add(mlpModel)
    }

//    model
//      .add(Embedding[T](itemCount + 1, itemEmbed, init = "normal", inputLength = maxLength))
//      .add(GRU[T](200, returnSequences = true))
//      .add(GRU[T](200, returnSequences = false))
//      .add(Dense[T](numClasses, activation = "log_softmax"))

    val rnnModel = rnn
//      .add(AddConstant(1))
      .add(LookupTable[T](itemCount, itemEmbed))
      .add(Recurrent[T]().add(GRU(itemEmbed, 200)))
      .add(Select(2, -1))

    val srModel = Concat(2).add(ncf).add(rnnModel)
    model.add(srModel).add(Linear(mfEmbed + hiddenLayers.last + 200, numClasses))

    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object SessionRecommender {

  def train(
             model: Sequential[Float],
             train: RDD[Sample[Float]],
             inputDir: String,
             rnnName: String,
             logDir: String,
             maxEpoch: Int,
             batchSize: Int
           ): Module[Float] = {

    val split = train.randomSplit(Array(0.8, 0.2), 100)
    val trainRDD = split(0)
    val testRDD = split(1)

    model.summary()
    println("trainRDD count = " + trainRDD.count())

    val optimizer = Optimizer(
      model = model,
      sampleRDD = trainRDD,
      criterion = new SparseCategoricalCrossEntropy[Float](logProbAsInput = true, zeroBasedLabel = false),
      batchSize = batchSize
    )

    val trained_model = optimizer
      .setOptimMethod(new RMSprop[Float]())
      .setValidation(Trigger.everyEpoch, testRDD, Array(new Top5Accuracy[Float]()), batchSize)
      .setEndWhen(Trigger.maxEpoch(maxEpoch))
      .optimize()

    trained_model.saveModule(inputDir + rnnName + "Keras", null, overWrite = true)
    println("Model has been saved")

    trained_model
  }

  def predict(
             path: String,
             testRDD: RDD[Sample[Float]]
             ) = {
    val model = Module.loadModule[Float](path)
    model.predictClass(testRDD).foreach(
      println(_)
    )
  }
}

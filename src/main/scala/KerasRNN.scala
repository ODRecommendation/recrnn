import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import org.apache.spark.rdd.RDD

/**
  * Created by luyangwang on Dec, 2018
  *
  */
class KerasRNN {

  def buildModel(
                  numClasses: Int,
                  skuCount: Int,
                  maxLength: Int,
                  embedOutDim: Int
                ): Sequential[Float] = {
    val model = Sequential[Float]()

    model.add(Embedding[Float](skuCount + 1, embedOutDim, init = "normal", inputLength = maxLength))
      .add(GRU[Float](200, returnSequences = true))
      .add(GRU[Float](200, returnSequences = false))
      .add(Dense[Float](numClasses, activation = "log_softmax"))
    model
  }

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
}

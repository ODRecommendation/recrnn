import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.CrossEntropyCriterion
import com.intel.analytics.bigdl.nn.keras._
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.optim._
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

    model.add(Embedding(skuCount + 1, embedOutDim, inputShape = Shape(5))).setName("Embedding")
        .add(GRU(200, returnSequences = true)).setName("GRU1")
        .add(GRU(200, returnSequences = false)).setName("GRU1")
        .add(Dropout(0.2))
        .add(Dense(numClasses))
//        .add(Activation("softmax"))
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
    val optimizer = Optimizer(
      model = model,
      sampleRDD = trainRDD,
//      criterion = new SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false),
      criterion = new CrossEntropyCriterion[Float](),
      batchSize = batchSize
    )

    val trained_model = optimizer
      .setOptimMethod(new RMSprop[Float]())
//      .setOptimMethod(new Adagrad[Float]())
//      .setTrainSummary(new TrainSummary(logDir, "recRNNTrainingSum"))
//      .setValidationSummary(new ValidationSummary(logDir, "recRNNValidationSum"))
//      .setCheckpoint(modelPath, Trigger.everyEpoch)
      .setValidation(Trigger.everyEpoch, testRDD, Array(new Top5Accuracy[Float]()), batchSize)
      .setEndWhen(Trigger.maxEpoch(maxEpoch))
      .optimize()

    trained_model.saveModule(inputDir + rnnName + "Keras", null, overWrite = true)
    println("Model has been saved")

    trained_model
  }
}

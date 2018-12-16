import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.{Sequential => _}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import org.apache.spark.rdd.RDD
 /**
  * Created by luyangwang on Dec, 2018
  *
  */
class RecRNN {

  def buildModel(
                  numClasses: Int,
                  skuCount: Int,
                  maxLength: Int,
                  embedOutDim: Int
                ): Sequential[Float] = {
    val model = Sequential[Float]()

    model
      .add(LookupTable[Float](skuCount, embedOutDim)).setName("Embedding")
      .add(Recurrent[Float]().add(GRU(embedOutDim, 200))).setName("GRU")
      .add(Dropout(0.2))
      .add(Select(2, -1))
      .add(Linear[Float](200, numClasses)).setName("Linear")
      .add(LogSoftMax())

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
           ):Module[Float] = {

    val split = train.randomSplit(Array(0.8, 0.2), 100)
    val trainRDD = split(0)
    val testRDD = split(1)
    val optimizer = Optimizer(
      model = model,
      sampleRDD = trainRDD,
      criterion = new CrossEntropyCriterion[Float](),
      batchSize = batchSize
    )

    val trained_model = optimizer
      .setOptimMethod(new RMSprop[Float]())
//      .setTrainSummary(new TrainSummary(logDir, "recRNNTrainingSum"))
//      .setValidationSummary(new ValidationSummary(logDir, "recRNNValidationSum"))
//      .setCheckpoint(modelPath, Trigger.everyEpoch)
      .setValidation(Trigger.everyEpoch, testRDD, Array(new Top5Accuracy[Float]()), batchSize)
      .setEndWhen(Trigger.maxEpoch(maxEpoch))
      .optimize()

    trained_model.saveModule(inputDir + rnnName, null, overWrite = true)
    println("Model has been saved")

    trained_model
  }
}

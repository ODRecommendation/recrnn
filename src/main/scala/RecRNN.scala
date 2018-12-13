import breeze.linalg.{max, min}
import breeze.numerics.{floor, pow}
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.{Sequential => _, _}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
//import com.intel.analytics.bigdl.nn.keras._
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table
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
    val branches = Concat[Float](2)
//    val embedOutDim = max(floor(pow(skuCount,1 / 4)), 10)
//    println("embedOutDim: " + embedOutDim.toString)

    var embedWidth = 0
    (1 to maxLength).foreach { _ =>
      val lookupTable = LookupTable[Float](skuCount, embedOutDim)
      lookupTable.setWeightsBias(
        Array(Tensor[Float](skuCount, embedOutDim).randn(0, 0.1)))
      branches.add(
        Sequential[Float]()
          .add(Select[Float](2, 1 + embedWidth))
          .add(AddConstant(1.0))
          .add(lookupTable))
          .setName(s"branch${embedWidth + 1}")
      embedWidth += 1
    }

    model
      .add(branches)
      .add(BiRecurrent[Float](JoinTable[Float](2, 2).asInstanceOf[AbstractModule[Table, Tensor[Float], Float]]).add(GRU(embedOutDim, 200)))
      .add(Dropout(0.2))
      .add(Select(2, -1))
      .add(Linear[Float](400, numClasses))
      .add(LogSoftMax())

    model
  }

  def train(
             model: Sequential[Float],
             train: RDD[Sample[Float]],
             modelPath: String,
             maxEpoch: Int,
             batchSize: Int
           ):Module[Float] = {

    val split = train.randomSplit(Array(0.7, 0.3), 100)
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
//      .setTrainSummary(new TrainSummary("./modelFiles", "recRNNTrainingSum"))
//      .setValidationSummary(new ValidationSummary("./modelFiles", "recRNNValidationSum"))
      .setValidation(Trigger.maxEpoch(maxEpoch), testRDD, Array(new Top1Accuracy[Float]()), batchSize)
      .setEndWhen(Trigger.maxEpoch(maxEpoch))
      .optimize()

    trained_model.saveModule(modelPath, null, overWrite = true)
    println("model has been saved")

    trained_model
  }
}

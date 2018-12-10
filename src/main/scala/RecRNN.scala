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
 /**
  * Created by luyangwang on Dec, 2018
  *
  */
class RecRNN {

  def buildModel(numClasses: Int, skuCount: Int): Sequential[Double] = {
    val model = Sequential[Double]()

    model
      .add(AddConstant(1))
      .add(LookupTable[Double](skuCount, 300))
      .add(TemporalConvolution(300, 250, 3))
      .add(ReLU())
      .add(TemporalMaxPooling(3))
      .add(Dropout(0.2))
      .add(BiRecurrent[Double](JoinTable[Double](2, 2).asInstanceOf[AbstractModule[Table, Tensor[Double], Double]]).add(GRU(250, 200)))
      .add(Select(2, -1))
      .add(Linear[Double](400, numClasses))
      .add(LogSoftMax())

//    model
////        .add(Flatten(Shape(Array(1, 5, 98))))
//        .add(Embedding(skuCount, 200, inputShape = Shape(10)))
//        .add(GRU(200))
//        .add(Dense(numClasses, activation = "sigmoid"))

    model
  }

  def train(
             model: Sequential[Double],
             train: RDD[Sample[Double]],
             modelPath: String,
             maxEpoch: Int,
             batchSize: Int
           ):Module[Double] = {

    val rmsp = new RMSprop[Double]()
    val adagrad = new Adagrad[Double]()
//    val loss = new ClassNLLCriterion[Double]()
    val loss = new CrossEntropyCriterion[Double]()


//      model.compile(
//        rmsp,
//        loss
//  //      Array(new Loss[Float]().asInstanceOf[ValidationMethod[Float]])
//      )
//      model.fit(train, batchSize)

    val split = train.randomSplit(Array(0.8, 0.2), 100)
    val trainRDD = split(0)
    val testRDD = split(1)
    val optimizer = Optimizer(
      model = model,
      sampleRDD = trainRDD,
      criterion = loss,
      batchSize = batchSize
    )

    val trained_model = optimizer
      .setOptimMethod(rmsp)
//      .setTrainSummary(new TrainSummary("./modelFiles", "recRNNTrainingSum"))
//      .setValidationSummary(new ValidationSummary("./modelFiles", "recRNNValidationSum"))
      .setValidation(Trigger.everyEpoch, testRDD, Array(new Top5Accuracy[Double]()), batchSize)
      .setEndWhen(Trigger.maxEpoch(maxEpoch))
      .optimize()

    trained_model.saveModule(modelPath, null, overWrite = true)

    println("model has been saved")

    trained_model
  }
}

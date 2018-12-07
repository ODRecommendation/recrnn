package model

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
//import com.intel.analytics.bigdl.nn.keras._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, Table}
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.nn._

/**
  * Created by luyangwang on Dec, 2018
  *
  */
class RecRNN {

  def buildModel(numClasses: Int) = {
    val model = Sequential[Double]()

    model
//      .add(Select(2, -1))
//      .add(LookupTable[Double](200, 100))
      .add(TemporalConvolution(200, 150, 3))
      .add(ReLU())
      .add(TemporalMaxPooling(3))
      .add(Dropout(0.2))
      .add(BiRecurrent[Double](JoinTable[Double](2, 2).asInstanceOf[AbstractModule[Table, Tensor[Double], Double]]).add(GRU(150, 100)))
//      .add(Recurrent[Double]().add(GRU(200, 100)))
      .add(Select(2, -1))
      .add(Linear[Double](200, numClasses))
      .add(LogSoftMax())

//    model
//        .add(Flatten(Shape(Array(1, 5, 98))))
//        .add(Embedding(98, 50))
//        .add(GRU(50))
//        .add(Dense(numClasses, "sigmoid"))

    model
  }

  def train(
             model: Sequential[Double],
             train: RDD[Sample[Double]],
             modelPath: String,
             batchSize: Int
           ) = {

    val rmsp = new RMSprop[Double]()
    val adagrad = new Adagrad[Double]()
    val loss = new ClassNLLCriterion[Double]()

    //    val loss = new BCECriterion[Float]()
    //    model.compile(
    //      rmsp,
    //      loss
    ////      Array(new Loss[Float]().asInstanceOf[ValidationMethod[Float]])
    //    )
    //    model.fit(train, batchSize)

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
      //      .setTrainSummary(new TrainSummary("./modelFiles", "stemCellTrainingSum"))
      //      .setValidationSummary(new ValidationSummary("./modelFiles", "stemCellValidationSum"))
      .setValidation(Trigger.everyEpoch, testRDD, Array(new Top1Accuracy[Double]()), batchSize)
      .setEndWhen(Trigger.maxEpoch(3))
      .optimize()

    trained_model.saveModule(modelPath, null, true)

    //    putS3Obj(bucketName, "stemCell/ModelFiles/commentRNN", modelPath)
    println("model has been saved")

    trained_model
  }
}

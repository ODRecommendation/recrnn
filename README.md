# RNN RECOMMENDATION USING BIGDL AND SPARK
A distributed rnn product recommendation implementation using BigDL and Spark. This recommendation algorithm only requires sequence of item agent has interacted as input therefore works with product recommendation for both identified / anonymous agent.

## Requirements
```scala
val sparkVersion = "2.3.1"
val analyticsZooVersion = "0.3.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion
)
libraryDependencies += "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.7.1-spark_2.3.1" % analyticsZooVersion
```

## How to use
Run model on your own data by replacing below value, if leave them blank model will run on provided sample data under modelFiles folder
```scala
    val params = ModelParams(
      maxLength = 10,
      maxEpoch = 10,
      batchSize = 2560,
      embedOutDim = 200,
      inputDir = "./modelFiles/",
      logDir = "./log/",
      dataName = "recrnn.csv",
      stringIndexerName = "skuIndexer",
      rnnName = "rnnModel"
    )
```
Package code to one jar and run as spark job
```scala
sbt -J-Xmx2G assembly
spark-submit --class Main ${location of assembled jar}
```
If you need to save the output file to AWS S3 bucket, simply change inputDir to your S3 path add below code to upload stringIndexerMleap model to your bucket, see below example
```scala
putS3Obj(
      bucketName = "your bucketName",
      fileKey = "path to your folder",
      filePath = currentDir + params.stringIndexerName + ".zip"
    )
```

## Contact & Feedback

 If you encounter bugs, feel free to submit an issue or pull request.
 Also you can mail to:
 * Luyang, Wang (tmacraft@hotmail.com)
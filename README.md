# RNN RECOMMENDATION USING BIGDL AND SPARK
A distributed rnn product recommendation implementation using BigDL and Spark. This recommendation can consider both customer's previous purchase history and current session behavior to cpture the latest purchase intent.

## Requirements
```scala
resolvers += "ossrh repository" at "https://oss.sonatype.org/content/repositories/snapshots/"

// set the main class for 'sbt run'
mainClass := Some("Main")

val sparkVersion = "2.4.0"
val bigDLVersion = "0.7.2"
val analyticsZooVersion = "0.4.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion
)
libraryDependencies += "com.intel.analytics.zoo" % s"analytics-zoo-bigdl_$bigDLVersion-spark_$sparkVersion" % analyticsZooVersion
libraryDependencies += "ml.combust.mleap" %% "mleap-spark" % "0.12.0"
libraryDependencies += "ml.combust.mleap" %% "mleap-spark-extension" % "0.12.0"
libraryDependencies += "com.amazonaws" % "aws-java-sdk" % "1.11.354"
libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.5"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.5" % "test"
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

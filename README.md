# RNN RECOMMENDATION USING BIGDL AND SPARK
A distributed rnn product recommendation algorithm using BigDL and Spark. This recommendation algorithm only requires sequence of item agent has interacted as input therefore works with both identified / anonymous agent product recommendation.

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

## How to run example
```scala
sbt assembly
spark-submit --class Main ${location of assembled jar}

```

## Contact & Feedback

 If you encounter bugs, feel free to submit an issue or pull request.
 Also you can mail to:
 * Luyang, Wang (tmacraft@hotmail.com)
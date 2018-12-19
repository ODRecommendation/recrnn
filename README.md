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
1. Run exmaple

```scala
sbt assembly
spark-submit --class Main ${location of assembled jar}

```

## Contact & Feedback

 If you encounter bugs, feel free to submit an issue or pull request.
 Also you can mail to:
 * Luyang, Wang (tmacraft@hotmail.com)
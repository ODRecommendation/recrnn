name := "recRNN"

version := "0.1"

scalaVersion := "2.11.12"

// set the main class for 'sbt run'
mainClass := Some("Main")

val sparkVersion = "2.3.1"
val analyticsZooVersion = "0.3.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion
)
libraryDependencies += "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.7.1-spark_2.3.1" % analyticsZooVersion
libraryDependencies += "ml.combust.mleap" %% "mleap-spark" % "0.12.0"
libraryDependencies += "ml.combust.mleap" %% "mleap-spark-extension" % "0.12.0"
libraryDependencies += "com.amazonaws" % "aws-java-sdk" % "1.11.354"

assemblyMergeStrategy in assembly := {
  case manifest if manifest.contains("MANIFEST.MF") =>
    // We don't need manifest files since sbt-assembly will create
    // one with the given settings
    MergeStrategy.discard
  case referenceOverrides if referenceOverrides.contains("reference-overrides.conf") =>
    // Keep the content for all reference-overrides.conf files
    MergeStrategy.concat
  case PathList("javax", "servlet", xs@_*) => MergeStrategy.first
  case PathList(ps@_*) if ps.last endsWith ".class" => MergeStrategy.first
  case PathList(ps@_*) if ps.last endsWith ".properties" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".html" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".proto" => MergeStrategy.first
  //  case PathList("org", "slf4j", xs@_*) => MergeStrategy.deduplicate
  case "application.conf" => MergeStrategy.concat
  case "unwanted.txt" => MergeStrategy.discard
  case PathList("META-INF", "services", "org.apache.hadoop.fs.FileSystem") => MergeStrategy.filterDistinctLines
  case x =>
    // For all the other files, use the default sbt-assembly merge strategy
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}
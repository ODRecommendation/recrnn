name := "recRNN"

version := "0.1"

scalaVersion := "2.11.12"

val sparkVersion = "2.3.1"
val analyticsZooVersion = "0.3.0"

mainClass in Compile := Some("main.Main")
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion
)
libraryDependencies += "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.7.1-spark_2.3.1" % analyticsZooVersion

assemblyMergeStrategy in assembly := {
  case manifest if manifest.contains("MANIFEST.MF") =>
    // We don't need manifest files since sbt-assembly will create
    // one with the given settings
    MergeStrategy.discard
  case referenceOverrides if referenceOverrides.contains("reference-overrides.conf") =>
    // Keep the content for all reference-overrides.conf files
    MergeStrategy.concat
  case x =>
    // For all the other files, use the default sbt-assembly merge strategy
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}

assemblyMergeStrategy in assembly := {
  case PathList("javax", "servlet", xs@_*) => MergeStrategy.first
  case PathList(ps@_*) if ps.last endsWith ".class" => MergeStrategy.first
  case PathList(ps@_*) if ps.last endsWith ".css" => MergeStrategy.first
  case PathList(ps@_*) if ps.last endsWith ".properties" => MergeStrategy.first
  case PathList(ps@_*) if ps.last endsWith ".proto" => MergeStrategy.last
  case "application.conf" => MergeStrategy.concat
  case "unwanted.txt" => MergeStrategy.discard
  case PathList("META-INF", "services", "org.apache.hadoop.fs.FileSystem") => MergeStrategy.filterDistinctLines
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}
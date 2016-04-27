name := "Assignment11"

version := "0.0.1"

scalaVersion := "2.10.4"

// additional libraries
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.1.0" % "provided"
)
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-mllib" % "1.3.0" % "provided"
)
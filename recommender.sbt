name := "Movie Recommender"

version := "0.0.1"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.3.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.3.0"
// libraryDependencies += "org.apache.spark" %% "spark-ml" % "1.3.0"

resolvers += "Akka Repository" at "http://repo.akka.io/releases/"
resolvers += "twttr.com" at "http://maven.twttr.com/"

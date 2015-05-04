
package edu.cornell.tech.cs5304.lab3

import java.io.File

import scala.io.Source

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}
// import org.apache.spark.rdd.RDD
// import org.apache.spark.mllib.linalg._
// import org.apache.spark.mllib.feature._



// import scala.math.sqrt
// import scala.math.pow
import java.io._



// import org.apache.hadoop.mapred.SequenceFileInputFormat
// import org.apache.hadoop.io.{LongWritable, Text}
 
// import com.esotericsoftware.kryo.Kryo
// import org.apache.spark.serializer.KryoRegistrator
 
// class Registrator extends KryoRegistrator {
//   override def registerClasses(kryo: Kryo) {
//     kryo.register(classOf[LongWritable])
//     kryo.register(classOf[Text])
//   }
// }

object MovieRecommender {
  /* find ngrams that match a regex; args are regex output input [input ..] */

  def main(args: Array[String]) {

  	// val writer = new PrintWriter(new File(args(0)))
  	// def writeln(str: String): Unit = writer.write(str + '\n')

    val conf = new SparkConf()
      .setAppName("movieRecommender")
      // .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      // .set("spark.kryo.registrator", "Registrator")
    val sc = new SparkContext(conf)

    val ratingsFile = args(0)
    val developmentMode = true

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    // val outputDirectory = args(1)
    // val filePrefix = args(2)


    // Load and parse the data
    // UserID::MovieID::Rating::Timestamp
    val data = sc.textFile(ratingsFile)
    val ratings: RDD[(Long, Rating)] = {
        val totalData = 
            data
                .map(_.split("::") match { case Array(userId, movieId, rating, timestamp) =>
                    (timestamp.toLong, Rating(userId.toInt, movieId.toInt, rating.toDouble))
                })
        if(developmentMode) {
            val subsetOfData = 
            totalData
                .filter(pair => {
                    val filterVal = (pair._1 / 47) % 1000
                    filterVal < 1 
                })
            subsetOfData

        }
        else totalData
    }

    println("There are " + ratings.count + " ratings")

    val sampledDataWeights = Array(0.20, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08)

    val dataSets: Array[RDD[Rating]] = ratings.values.randomSplit(sampledDataWeights)

    val testSet: RDD[Rating]= dataSets(0)
    val xvalidationSets: Array[RDD[Rating]] = dataSets.slice(1, dataSets.length)

    // create (Model, MSE) pairs for each xvalidation set

    def computeMSE(model: MatrixFactorizationModel, data: RDD[Rating]): Double = {
        
        val usersMovies = data.map { case Rating(user, movie, rating) =>
          (user, movie)
        }

        val predictions = 
          model.predict(usersMovies).map { case Rating(user, movie, rating) => 
            ((user, movie), rating)
          }
        
        val ratesAndPreds = data.map { case Rating(user, movie, rating) => 
          ((user, movie), rating)
        }.join(predictions)
        
        val MSE = ratesAndPreds.map { case ((user, movie), (rating1, rating2)) => 
          val err = (rating1 - rating2)
          err * err
        }.mean()

        println("Mean Squared Error = " + MSE)
        MSE
    }

    // val modelMSEPairs: Array[(MatrixFactorizationModel, Double)] = 
    //     for{
    //         i <- (0 until xvalidationSets.length).toArray
    //     } yield {
    //         val validationSet: RDD[Rating] = xvalidationSets(i)
    //         val trainingSet: RDD[Rating] = 
    //             (0 until xvalidationSets.length)
    //             .filter(j => i != j)
    //             .map(j => xvalidationSets(j))
    //             .reduce(_ ++ _)

    //         val als = new ALS()
    //         als.setIterations(1)
    //         val model: MatrixFactorizationModel = als.run(trainingSet)

    //         val mse = computeMSE(model, validationSet)

    //         (model, mse)
    //     }

    // val bestModelPair = 
    //     modelMSEPairs
    //     .sortBy(pair => pair._2)
    //     .apply(0)

    val parameters: (Int, Double, Int) = {
        val ranks = List(8, 12)
        val lambdas = List(1.0, 10.0)
        val numIters = List(10, 20)
        val MSEParametersPairs: List[(Double, (Int, Double, Int))] = for {
            rank <- ranks
            lambda <- lambdas
            iter <- numIters
        } yield {
            val modelMSEPairs = 
                for {
                    i <- (0 until xvalidationSets.length).toArray
                } yield {
                    val validationSet: RDD[Rating] = xvalidationSets(i)
                    val trainingSet: RDD[Rating] = 
                        (0 until xvalidationSets.length)
                        .filter(j => i != j)
                        .map(j => xvalidationSets(j))
                        .reduce(_ ++ _)

                    val als = new ALS()
                    als.setRank(rank)
                    als.setLambda(lambda)
                    als.setIterations(iter)
                    println("Training: Iteration " + i + " with rank=" + rank + ", lambda=" + lambda + ", iters=" + iter)
                    val model: MatrixFactorizationModel = als.run(trainingSet)

                    val mse = computeMSE(model, validationSet)

                    (model, mse)
                }
            val totalMSE = 
                modelMSEPairs
                .map(pair => pair._2)
                .sum

            (totalMSE / xvalidationSets.length.toDouble, (rank, lambda, iter))
        }

        val bestPair = 
            MSEParametersPairs
            .sortBy(pair => pair._1)
            .head

        bestPair._2
    }

    val rank = parameters._1
    val lambda = parameters._2
    val iter = parameters._3
    val trainingSet = xvalidationSets.reduce(_ ++ _)

    val als = new ALS()
    als.setRank(rank)
    als.setLambda(lambda)
    als.setIterations(iter)
    val bestModel: MatrixFactorizationModel = als.run(trainingSet)

    val testMSE = computeMSE(bestModel, testSet)
    println("The final MSE is " + testMSE)
  }
}
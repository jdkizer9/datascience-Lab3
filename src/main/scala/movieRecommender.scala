
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
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import java.io._
import scala.io._

object MovieRecommender {
  /* find ngrams that match a regex; args are regex output input [input ..] */

  def main(args: Array[String]) {

    val conf = new SparkConf()
      .setAppName("movieRecommender")
      // .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      // .set("spark.kryo.registrator", "Registrator")
    val sc = new SparkContext(conf)

    val ratingsFile = args(0)

    val developmentMode1 = false
    val developmentMode2 = false
    val developmentMode3 = false

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)


    // Load and parse the data
    // UserID::MovieID::Rating::Timestamp
    val data = sc.textFile(ratingsFile)
    val ratings: RDD[(Long, Rating)] = {
        val totalData = 
            data
                .map(_.split("::") match { case Array(userId, movieId, rating, timestamp) =>
                    (timestamp.toLong, Rating(userId.toInt, movieId.toInt, rating.toDouble))
                })
        if(developmentMode1) {
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

    // val sampledDataWeights = Array(0.20, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08)
    val sampledDataWeights = 
        if(developmentMode2) Array(0.20, 0.40, 0.40) 
        // else Array(0.20, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08)
        else Array(0.20, 0.20, 0.20, 0.20, 0.20)

    val dataSets: Array[RDD[Rating]] = ratings.values.randomSplit(sampledDataWeights)

    val testSet: RDD[Rating]= dataSets(0)
    val xvalidationSets: Array[RDD[Rating]] = dataSets.slice(1, dataSets.length)

    // create (Model, MSE) pairs for each xvalidation set

    def computeALS_RMSE(model: MatrixFactorizationModel, data: RDD[Rating]): Double = {
        
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

        println("Unable to predict " + (data.count - ratesAndPreds.count) + " ratings")
        
        val MSE = ratesAndPreds.map { case ((user, movie), (rating1, rating2)) => 
          val err = (rating1 - rating2)
          err * err
        }.mean()

        val RMSE = Math.sqrt(MSE)

        println("Root Mean Squared Error = " + RMSE)
        RMSE
    }

    def computeLinear_RMSE(model: LinearRegressionModel, data: RDD[LabeledPoint]): Double = {
        val features: RDD[Vector] = data.map { case LabeledPoint(label, features) => features}

        val predictions:RDD[(Long, Double)] = 
            model.predict(features).zipWithIndex.map(pair => (pair._2, pair._1))

        val ratesAndPreds: RDD[(Long, (Double, Double))] = 
            data.map { case LabeledPoint(label, features) => label}
            .zipWithIndex.map(pair => (pair._2, pair._1))
            .join(predictions)

        // ratesAndPreds.foreach(println)
        val MSE = ratesAndPreds.map { case (id, (rating1, rating2)) => 
          val err = (rating1 - rating2)
          err * err
        }.mean()

        val RMSE = Math.sqrt(MSE)

        println("Root Mean Squared Error = " + RMSE)
        RMSE
    }

    def meanSelection(feature: Vector): Double ={
        val featureArray = feature.toArray
        val sortedSlice =
            featureArray
            .sorted
            .slice(featureArray.length/4, 3*featureArray.length/4)

        sortedSlice.sum / sortedSlice.length.toDouble
    }

    def medianSelection(feature: Vector): Double ={
        val featureArray = feature.toArray
        val sortedSlice =
            featureArray
            .sorted
            
        if((sortedSlice.length % 2) == 1) {
            sortedSlice(sortedSlice.length/2 + 1)
        } else {
            val v1 = sortedSlice(sortedSlice.length/2)
            val v2 = sortedSlice(sortedSlice.length/2 + 1)
            (v1 + v2) / 2
        }
    }

    def computeOtherModel_RMSE(mapper: (Vector) => Double, data: RDD[LabeledPoint]): Double = {
        val ratesAndPreds: RDD[(Double, Double)] = data.map { case LabeledPoint(label, features) => (label, mapper(features))}

        val MSE = ratesAndPreds.map { case (rating1, rating2) => 
          val err = (rating1 - rating2)
          err * err
        }.mean()

        val RMSE = Math.sqrt(MSE)

        println("Root Mean Squared Error = " + RMSE)
        RMSE
    }

    var allModels:Array[MatrixFactorizationModel] = Array[MatrixFactorizationModel]()

    val parameters: (Int, Double, Int) = {
        val ranks = 
            if(developmentMode3) List(8)
            else List(8, 12)

        val lambdas = 
            if(developmentMode3) List(1.0)
            else List(1.0, 10.0)

        val numIters = 
            if(developmentMode3) List(1)
            else List(10, 20)

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

                    println("Training: Iteration " + i + " with rank=" + rank + ", lambda=" + lambda + ", iters=" + iter)
                    val model: MatrixFactorizationModel = ALS.train(trainingSet, rank, iter, lambda)

                    val rmse = computeALS_RMSE(model, validationSet)

                    allModels = allModels :+ model

                    (model, rmse)
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

    // allModels.zipWithIndex.foreach(pair => {
    //     pair._1.save(sc, modelOutputDirectory + "individualModel_"+ pair._2 + ".mfm")
    // })

    val rank = parameters._1
    val lambda = parameters._2
    val iter = parameters._3
    val trainingSet: RDD[Rating] = xvalidationSets.reduce(_ ++ _)

    val bestModel: MatrixFactorizationModel = ALS.train(trainingSet, rank, iter, lambda)

    val testRMSE = computeALS_RMSE(bestModel, testSet)
    println("The final ALS Model RMSE is " + testRMSE)

    // bestModel.save(sc, modelOutputDirectory + "bestModel.mfm")


    def dataSetFromEnsemble(data: RDD[Rating]): RDD[((Int, Int), LabeledPoint)] = {
        

        val usersMovies = data.map { case Rating(user, movie, rating) =>
          (user, movie)
        }

        usersMovies.cache()

        // println("Number of ratings in dataset: " + usersMovies.count)

        var computedFeatures: RDD[((Int, Int), (Int, Double))] = data.context.emptyRDD

        for (j <- 0 until allModels.length) {

            // computedFeatures.cache()
            val model = allModels(j)

            // println("Number of features: " + usersMovies.count)
            val predictions = 
              model.predict(usersMovies).map { case Rating(user, movie, rating) => 
                ((user, movie), (j, rating))
              }

            // println("Number of predictions: " + predictions.count)

            // predictions.foreach(println)

            // val oldFeatures = computedFeatures
            computedFeatures = computedFeatures ++ predictions
            // oldFeatures.unpersist(blocking=false)
        }

        val groupedFeatureIterables: RDD[((Int, Int), Iterable[(Int, Double)])] = 
            computedFeatures
            .groupByKey

        // groupedFeatureIterables.foreach(pair => println(pair._1 + ": " + pair._2.size))

        val groupedFeatures: RDD[((Int, Int), Vector)] =
            groupedFeatureIterables
            .filter(pair => pair._2.size == allModels.length)
            .mapValues(it => {
                if(it.size != allModels.length){
                    println("the number of features are: " + it.size)
                    assert(false)
                }
                
                val featureArray = 
                it
                .toArray
                .sortBy(pair => pair._1)
                .map(pair => pair._2)

                assert(featureArray.length == allModels.length)
                Vectors.dense(featureArray)
            })

        val ratesAndPreds:RDD[((Int, Int), (Double, Vector))] = data.map { case Rating(user, movie, rating) => 
          ((user, movie), rating)
        }.join(groupedFeatures)

        // ratesAndPreds.foreach(println)


        val outputPoints: RDD[((Int, Int), LabeledPoint)] = 
            ratesAndPreds
            .mapValues(pair => new LabeledPoint(pair._1, pair._2))

        outputPoints
    }

    



    val convertedTrainingSet = dataSetFromEnsemble(trainingSet)
    convertedTrainingSet.cache()

    println("Training Set Conversion: " + convertedTrainingSet.count + "/" + trainingSet.count)

    // val numIterations = {
    //     val computedIterations = (10000000 / trainingSet.count) + 1
    //     if(computedIterations < 5) 5
    //     else computedIterations.toInt
    // }

    // convertedTrainingSet.foreach(println)

    val numIterations = 100
    val linearModel = LinearRegressionWithSGD.train(convertedTrainingSet.values.cache(), numIterations)

    val convertedTestSet = dataSetFromEnsemble(testSet)
    convertedTestSet.cache()

    // convertedTestSet.foreach(println)

    println("Test Set Conversion: " + convertedTestSet.count + "/" + testSet.count)
    val linearTestRMSE = computeLinear_RMSE(linearModel, convertedTestSet.values)
    println("The Linear Model RMSE is " + linearTestRMSE)

    val meanModelTestRMSE = computeOtherModel_RMSE(meanSelection, convertedTestSet.values)
    println("The Mean Model RMSE is " + meanModelTestRMSE)

    val medianModelTestRMSE = computeOtherModel_RMSE(medianSelection, convertedTestSet.values)
    println("The Median Model RMSE is " + medianModelTestRMSE)

    // linearModel.save(sc, modelOutputDirectory + "linearModel.lrm")

    // val sameModel = MatrixFactorizationModel.load(sc, "myModelPath")


  }
}
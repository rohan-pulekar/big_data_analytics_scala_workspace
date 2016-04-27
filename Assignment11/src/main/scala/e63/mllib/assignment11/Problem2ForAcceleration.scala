package e63.mllib.assignment11

import scala.collection.immutable.Map
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.classification.NaiveBayes

/**
 * @author Rohan Pulekar
 * Purpose of this program:
 * This program is for Assignment11 Problem2 (Acceleration) of e63 course (Big Data Analytics) of Spring 2016 batch of Harvard Extension School
 *
 */
object Problem2ForAcceleration {
  def main(args: Array[String]) {

    // create spark conf and spark context
    val conf = new SparkConf().setAppName("Assignment11-Problem2ForAcceleration")
    val sc = new SparkContext(conf)

    // take input file name from program parameter
    val inputFileName = args(0)

    // construct RDD from input file
    val rawCarDataRDDWithHeaders = sc.textFile(inputFileName)

    // get the headers row of input file
    val headersRow = rawCarDataRDDWithHeaders.first()

    // construct a headers map
    val headersMap: Map[String, Int] = Map("Record_num" -> 0, "Acceleration" -> 1,
      "Cylinders" -> 2, "Displacement" -> 3, "Horsepower" -> 4, "Manufacturer" -> 5, "Model" -> 6,
      "Model_Year" -> 7, "MPG" -> 8, "Origin" -> 9, "Weight" -> 10)

    // create an RDD of data without the headers
    val uncleanCarDataRDDOfUnsplitLines = rawCarDataRDDWithHeaders.filter(rowAsAString => (rowAsAString != headersRow))

    // split the data by comma
    val uncleanCarDataRDD = uncleanCarDataRDDOfUnsplitLines.map(rowAsAString => rowAsAString.split(","))

    // trim the data as it contains spaces
    val cleanCarDataRDD = uncleanCarDataRDD.map(rowAsArrayOfValues => rowAsArrayOfValues.map(value => value.trim()))

    // create categoriy features
    val manufacturerCategoriesRDD = cleanCarDataRDD.map(rowAsArrayOfValues => rowAsArrayOfValues(headersMap("Manufacturer"))).distinct().collect
    val originCategoriesRDD = cleanCarDataRDD.map(rowAsArrayOfValues => rowAsArrayOfValues(headersMap("Origin"))).distinct().collect

    // create categories map
    val categoriesMap = manufacturerCategoriesRDD.union(originCategoriesRDD).zipWithIndex.toMap
    val numberOfCategories = categoriesMap.size
    println("Number of categories:" + numberOfCategories)

    // construct the entire data RDD, also filter the NaN values
    val dataRDDForAcceleration = cleanCarDataRDD.map { rowAsArrayOfValues =>
      var label = 0.0
      if (rowAsArrayOfValues(headersMap("Acceleration")) != "NaN") {
        label = rowAsArrayOfValues(headersMap("Acceleration")).toDouble
      }

      val categoryFeatures = Array.ofDim[Double](numberOfCategories)

      val manufacturerCategoryIdx = categoriesMap(rowAsArrayOfValues(headersMap("Manufacturer")))
      categoryFeatures(manufacturerCategoryIdx) = 1.0

      val originCategoryIdx = categoriesMap(rowAsArrayOfValues(headersMap("Origin")))
      categoryFeatures(originCategoryIdx) = 1.0

      val nonCategoryFeatures = rowAsArrayOfValues.slice(2, 3).union(rowAsArrayOfValues.slice(3, 4)).union(rowAsArrayOfValues.slice(7, 8)).map(feature => feature.toDouble)

      val features = categoryFeatures ++ nonCategoryFeatures

      LabeledPoint(label, Vectors.dense(features))
    }

    // cache the RDD
    dataRDDForAcceleration.cache

    // create indexed RDD in order to split it into test and training data
    val dataRDDForAccelerationWithIdx = dataRDDForAcceleration.zipWithIndex().map(mapEntry => (mapEntry._2, mapEntry._1))

    // create test and training data RDDs (these will be with ids)
    val testDataRDDForAccelerationWithIdx = dataRDDForAccelerationWithIdx.sample(false, 0.1, 40)
    val trainDataRDDForAccelerationWithIdx = dataRDDForAccelerationWithIdx.subtract(testDataRDDForAccelerationWithIdx)

    // create test and training RDDs
    val testDataRDDForAcceleration = testDataRDDForAccelerationWithIdx.map(mapEntry => mapEntry._2)
    val trainingDataRDDForAcceleration = trainDataRDDForAccelerationWithIdx.map(mapEntry => mapEntry._2)

    println("Training data size:" + trainingDataRDDForAcceleration.collect().size)
    println("Test data size:" + testDataRDDForAcceleration.collect().size)
    println()

    // cache the training data RDD
    trainingDataRDDForAcceleration.cache

    // these are my settings for linear regression.
    // I tried a lot with different settings but these were the ones that worked for me
    val numIterationsForLR = 10000
    val stepSizeForLR = 0.0001

    /********************  Linear Regression with SGD model for Acceleration start ********************/
    val linearSGDTrainedModelForAcceleration = LinearRegressionWithSGD.train(trainingDataRDDForAcceleration, numIterationsForLR, stepSizeForLR)
    val linearSGDPredictedVsActualForAcceleration = testDataRDDForAcceleration.map { testDataRow =>
      (Math.round(linearSGDTrainedModelForAcceleration.predict(testDataRow.features)).toDouble, testDataRow.label)
    }
    println("Predicted vs Actual value of Acceleration for Linear Regression with SGD:")
    linearSGDPredictedVsActualForAcceleration.collect().foreach(println)
    var metrics = new RegressionMetrics(linearSGDPredictedVsActualForAcceleration)
    println("Performance metrics for Linear Regression with SGD Model for Acceleration:")
    printMetrics(metrics)
    println()
    /**********************  Linear Regression with SGD model for Acceleration end ********************/

    /********************  Random Forest model for Acceleration start ********************/
    val numTrees = 3
    val maxDepth = 4
    val maxBins = 9
    val randomForestTrainedModelForAcceleration = RandomForest.trainRegressor(trainingDataRDDForAcceleration, Map[Int, Int](),
      numTrees, "auto", "variance", maxDepth, maxBins)
    val randomForestPredictedVsActualForAcceleration = testDataRDDForAcceleration.map { testDataRow =>
      (Math.round(randomForestTrainedModelForAcceleration.predict(testDataRow.features)).toDouble, testDataRow.label)
    }
    metrics = new RegressionMetrics(randomForestPredictedVsActualForAcceleration)
    println("Performance metrics for Random Forest Model for Acceleration:")
    printMetrics(metrics)
    println()
    /********************  Random Forest model for Acceleration end ********************/

    /********************  Random Forest model for Gradient Boosted Model start ********************/
    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.numIterations = 10
    boostingStrategy.treeStrategy.maxDepth = 5
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    val gradientTrainedModelForAcceleration = GradientBoostedTrees.train(trainingDataRDDForAcceleration, boostingStrategy)
    val gradientPredictedVsActualForAcceleration = testDataRDDForAcceleration.map { testDataRow =>
      (Math.round(gradientTrainedModelForAcceleration.predict(testDataRow.features)).toDouble, testDataRow.label)
    }
    metrics = new RegressionMetrics(gradientPredictedVsActualForAcceleration)
    println("Performance metrics for Gradient Boosted Model for Acceleration:")
    printMetrics(metrics)
    println()
    /********************  Random Forest model for Gradient Boosted Model end ********************/

    
    // I was unable to scale the features with StandardScaler
    // If do that the linear regression model gives bad results.  So I decided not to
    // I was unable to use SVMWithSGD and LogisticRegressionWithSGD since label is not a boolean (0 or 1)

  }

  /**
   * function to print metrics
   */
  def printMetrics(regressionMetrics: RegressionMetrics): Unit = {
    println(s"MSE = ${regressionMetrics.meanSquaredError}")
    println(s"RMSE = ${regressionMetrics.rootMeanSquaredError}")
    println(s"R-squared = ${regressionMetrics.r2}")
    println(s"MAE = ${regressionMetrics.meanAbsoluteError}")
    println(s"Explained variance = ${regressionMetrics.explainedVariance}")
  }
}
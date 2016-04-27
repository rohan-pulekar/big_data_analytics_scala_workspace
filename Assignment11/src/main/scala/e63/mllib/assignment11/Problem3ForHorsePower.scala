package e63.mllib.assignment11

import scala.collection.immutable.Map
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree

/**
 * @author Rohan Pulekar
 * Purpose of this program:
 * This program is for Assignment11 Problem3 (Horse Power) of e63 course (Big Data Analytics) of Spring 2016 batch of Harvard Extension School
 *
 */
object Problem3ForHorsePower {
  def main(args: Array[String]) {

    // create spark conf and spark context
    val conf = new SparkConf().setAppName("Assignment11-Problem2ForHorsePower")
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

    // create category features
    val manufacturerCategoriesRDD = cleanCarDataRDD.map(rowAsArrayOfValues => rowAsArrayOfValues(headersMap("Manufacturer"))).distinct().collect
    val originCategoriesRDD = cleanCarDataRDD.map(rowAsArrayOfValues => rowAsArrayOfValues(headersMap("Origin"))).distinct().collect

    // create categories map
    val categoriesMap = manufacturerCategoriesRDD.union(originCategoriesRDD).zipWithIndex.toMap
    val numberOfCategories = categoriesMap.size
    println("Number of categories:" + numberOfCategories)

    // construct the entire data RDD, also filter the NaN values
    val dataRDDForHorsePower = cleanCarDataRDD.map { rowAsArrayOfValues =>
      var label = 0.0
      if (rowAsArrayOfValues(headersMap("Horsepower")) != "NaN") {
        label = rowAsArrayOfValues(headersMap("Horsepower")).toInt
      }

      val categoryFeatures = Array.ofDim[Double](numberOfCategories)

      val manufacturerCategoryIdx = categoriesMap(rowAsArrayOfValues(headersMap("Manufacturer")))
      categoryFeatures(manufacturerCategoryIdx) = 1.0

      val originCategoryIdx = categoriesMap(rowAsArrayOfValues(headersMap("Origin")))
      categoryFeatures(originCategoryIdx) = 1.0

      val nonCategoryFeatures = rowAsArrayOfValues.slice(2, 3).union(rowAsArrayOfValues.slice(3, 4)).union(rowAsArrayOfValues.slice(7, 8)).union(rowAsArrayOfValues.slice(10, 11)).map(feature => if (feature == "NaN") 0.0 else feature.toDouble)

      val features = categoryFeatures ++ nonCategoryFeatures

      LabeledPoint(label, Vectors.dense(features))
    }

    // cache the RDD
    dataRDDForHorsePower.cache

    val features = dataRDDForHorsePower.map(labeledPoint => labeledPoint.features)
    val featuresMatrix = new RowMatrix(features)
    val featuresMatrixSummary = featuresMatrix.computeColumnSummaryStatistics()
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(features)
    val scaledDataRDDForHorsePower = dataRDDForHorsePower.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))

    // create indexed RDD in order to split it into test and training data
    val dataRDDForHorsePowerWithIdx = scaledDataRDDForHorsePower.zipWithIndex().map(mapEntry => (mapEntry._2, mapEntry._1))

    // create test and training data RDDs (these will be with ids)
    val testDataRDDForHorsePowerWithIdx = dataRDDForHorsePowerWithIdx.sample(false, 0.1, 40)
    val trainDataRDDForHorsePowerWithIdx = dataRDDForHorsePowerWithIdx.subtract(testDataRDDForHorsePowerWithIdx)

    // create test and training RDDs
    val testDataRDDForHorsePower = testDataRDDForHorsePowerWithIdx.map(mapEntry => mapEntry._2)
    val trainingDataRDDForHorsePower = trainDataRDDForHorsePowerWithIdx.map(mapEntry => mapEntry._2)

    println("Training data size:" + trainingDataRDDForHorsePower.collect().size)
    println("Test data size:" + testDataRDDForHorsePower.collect().size)
    println()

    // cache the training data RDD
    trainingDataRDDForHorsePower.cache

    // these are my settings for decision tree.
    // I tried a lot with different settings but these were the ones that worked for me
    val maxDepth = 5
    val maxBins = 9

    /********************  Decision Tree model for Horse Power start ********************/
    val dTreeTrainedModelForHorsePower = DecisionTree.trainRegressor(trainingDataRDDForHorsePower, Map[Int, Int](), "variance", maxDepth, maxBins)
    val dTreePredictedVsActualForHorsePower = testDataRDDForHorsePower.map { testDataRow =>
      (Math.round(dTreeTrainedModelForHorsePower.predict(testDataRow.features)).toDouble, testDataRow.label)
    }
    println("Predicted vs Actual value of Horse Power for Decision Tree model")
    dTreePredictedVsActualForHorsePower.collect().foreach(println)
    println()
    var metrics = new RegressionMetrics(dTreePredictedVsActualForHorsePower)
    println("Performance metrics for Decision Tree Model for Horse Power:")
    printMetrics(metrics)
    println()
    /********************  Decision Tree model for Horse Power end ********************/

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
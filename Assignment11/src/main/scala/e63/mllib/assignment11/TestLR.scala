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

object TestLR {
  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local").setAppName("Assignment11-Problem2ForHorsePower")
    val sc = new SparkContext(conf)

    val rawCarDataRDDWithHeaders = sc.textFile("./data/Small_Car_Data_cleaned.csv")

    val headersRow = rawCarDataRDDWithHeaders.first()

    println("headers:")
    println(headersRow)
    println()

    val headersMap: Map[String, Int] = Map("Record_num" -> 0, "Acceleration" -> 1,
      "Cylinders" -> 2, "Displacement" -> 3, "Horsepower" -> 4, "Manufacturer" -> 5, "Model" -> 6,
      "Model_Year" -> 7, "MPG" -> 8, "Origin" -> 9, "Weight" -> 10)

    val uncleanCarDataRDDOfUnsplitLines = rawCarDataRDDWithHeaders.filter(rowAsAString => (rowAsAString != headersRow))

    val uncleanCarDataRDD = uncleanCarDataRDDOfUnsplitLines.map(rowAsAString => rowAsAString.split(","))

    val cleanCarDataRDD = uncleanCarDataRDD.map(rowAsArrayOfValues => rowAsArrayOfValues.map(value => value.trim()))

    val manufacturerCategoriesRDD = cleanCarDataRDD.map(rowAsArrayOfValues => rowAsArrayOfValues(headersMap("Manufacturer"))).distinct().collect
    val originCategoriesRDD = cleanCarDataRDD.map(rowAsArrayOfValues => rowAsArrayOfValues(headersMap("Origin"))).distinct().collect

    val categoriesMap = manufacturerCategoriesRDD.union(originCategoriesRDD).zipWithIndex.toMap
    val numberOfCategories = categoriesMap.size
    println("Number of categories:" + numberOfCategories)

    val dataRDDForHorsePower = cleanCarDataRDD.map { rowAsArrayOfValues =>

      val randomNum = scala.math.abs(scala.util.Random.nextInt()) % 200
      val delta = scala.math.abs(scala.util.Random.nextInt()) % 5

      var label = 0.0
      if (rowAsArrayOfValues(headersMap("Horsepower")) != "NaN") {
        label = rowAsArrayOfValues(headersMap("Horsepower")).toInt
      }

      val categoryFeatures = Array.ofDim[Double](numberOfCategories)

      val manufacturerCategoryIdx = categoriesMap(rowAsArrayOfValues(headersMap("Manufacturer")))
      categoryFeatures(manufacturerCategoryIdx) = 1.0

      val originCategoryIdx = categoriesMap(rowAsArrayOfValues(headersMap("Origin")))
      categoryFeatures(originCategoryIdx) = 1.0

      val nonCategoryFeatures = rowAsArrayOfValues.slice(2, 3).union(rowAsArrayOfValues.slice(3, 4)).union(rowAsArrayOfValues.slice(7, 8)).map(feature => feature.toDouble)

      val features = nonCategoryFeatures //categoryFeatures ++ nonCategoryFeatures

      LabeledPoint(label, Vectors.dense(features))
    }

    dataRDDForHorsePower.cache

    dataRDDForHorsePower.collect().foreach { println }

    val features = dataRDDForHorsePower.map(labeledPoint => labeledPoint.features)
    val featuresMatrix = new RowMatrix(features)
    val featuresMatrixSummary = featuresMatrix.computeColumnSummaryStatistics()
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(features)
    val scaledDataRDDForHorsePower = dataRDDForHorsePower.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))

    val dataRDDForHorsePowerWithIdx = dataRDDForHorsePower.zipWithIndex().map(mapEntry => (mapEntry._2, mapEntry._1))

    val testDataRDDForHorsePowerWithIdx = dataRDDForHorsePowerWithIdx.sample(false, 0.1, 40)
    val trainDataRDDForHorsePowerWithIdx = dataRDDForHorsePowerWithIdx.subtract(testDataRDDForHorsePowerWithIdx)

    val testDataRDDForHorsePower = testDataRDDForHorsePowerWithIdx.map(mapEntry => mapEntry._2)
    val trainingDataRDDForHorsePower = trainDataRDDForHorsePowerWithIdx.map(mapEntry => mapEntry._2)

    println("Test data size:" + testDataRDDForHorsePower.collect().size)
    println("Training data size:" + trainingDataRDDForHorsePower.collect().size)
    println()

    trainingDataRDDForHorsePower.cache
    
    val numIterationsForLR = 10000
    val stepSizeForLR = 0.001

    /*****  Linear Regression with SGD model for Horse Power start ****/
    val linearSGDTrainedModelForHorsePower = LinearRegressionWithSGD.train(trainingDataRDDForHorsePower, numIterationsForLR, stepSizeForLR)
    val linearSGDPredictedVsActualForHorsePower = testDataRDDForHorsePower.map { testDataRow =>
      (Math.round(linearSGDTrainedModelForHorsePower.predict(testDataRow.features)).toDouble, testDataRow.label)
    }
    println("Predicted vs Actual value of Horse Power for Linear Regression with SGD")
    linearSGDPredictedVsActualForHorsePower.collect().foreach(println)
    var metrics = new RegressionMetrics(linearSGDPredictedVsActualForHorsePower)
    println("Performance metrics for Linear Regression with SGD Model for Horse Power:")
    printMetrics(metrics)
    println()
    /*****  Linear Regression with SGD model for Horse Power end ****/

  }

  def printMetrics(regressionMetrics: RegressionMetrics): Unit = {
    println(s"MSE = ${regressionMetrics.meanSquaredError}")
    println(s"RMSE = ${regressionMetrics.rootMeanSquaredError}")
    println(s"R-squared = ${regressionMetrics.r2}")
    println(s"MAE = ${regressionMetrics.meanAbsoluteError}")
    println(s"Explained variance = ${regressionMetrics.explainedVariance}")
  }
}
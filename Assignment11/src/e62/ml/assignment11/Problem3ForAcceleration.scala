package e62.ml.assignment11

import scala.collection.immutable.Map

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.BoostingStrategy

object Problem3ForAcceleration {
  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local").setAppName("Assignment11-Problem2ForAcceleration")
    val sc = new SparkContext(conf)

    val rawCarDataRDDWithHeaders = sc.textFile("./data/Small_Car_Data.csv")

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

    val cylindersCategoriesRDD = cleanCarDataRDD.map(rowAsArrayOfValues => rowAsArrayOfValues(headersMap("Cylinders"))).distinct().collect
    val manufacturerCategoriesRDD = cleanCarDataRDD.map(rowAsArrayOfValues => rowAsArrayOfValues(headersMap("Manufacturer"))).distinct().collect
    val modelYearCategoriesRDD = cleanCarDataRDD.map(rowAsArrayOfValues => rowAsArrayOfValues(headersMap("Model_Year"))).distinct().collect
    val originCategoriesRDD = cleanCarDataRDD.map(rowAsArrayOfValues => rowAsArrayOfValues(headersMap("Origin"))).distinct().collect

    val categoriesMap = cylindersCategoriesRDD.union(manufacturerCategoriesRDD).union(modelYearCategoriesRDD).union(originCategoriesRDD).zipWithIndex.toMap
    val numberOfCategories = categoriesMap.size
    println("Number of categories:" + numberOfCategories)

    val dataRDDForAcceleration = cleanCarDataRDD.map { rowAsArrayOfValues =>
      var label = 0.0
      if (rowAsArrayOfValues(headersMap("Acceleration")) != "NaN") {
        label = rowAsArrayOfValues(headersMap("Acceleration")).toDouble
      }

      val categoryFeatures = Array.ofDim[Double](numberOfCategories)

      val cylindersCategoryIdx = categoriesMap(rowAsArrayOfValues(headersMap("Cylinders")))
      categoryFeatures(cylindersCategoryIdx) = 1.0

      val manufacturerCategoryIdx = categoriesMap(rowAsArrayOfValues(headersMap("Manufacturer")))
      categoryFeatures(manufacturerCategoryIdx) = 1.0

      val modelYearCategoryIdx = categoriesMap(rowAsArrayOfValues(headersMap("Model_Year")))
      categoryFeatures(modelYearCategoryIdx) = 1.0

      val originCategoryIdx = categoriesMap(rowAsArrayOfValues(headersMap("Origin")))
      categoryFeatures(originCategoryIdx) = 1.0

      val nonCategoryFeatures = rowAsArrayOfValues.slice(3, 4).union(rowAsArrayOfValues.slice(10, 12)).map(feature => if (feature == "NaN") 0.0 else feature.toDouble)

      val features = categoryFeatures ++ nonCategoryFeatures

      LabeledPoint(label, Vectors.dense(features))
    }

    dataRDDForAcceleration.cache

    val features = dataRDDForAcceleration.map(labeledPoint => labeledPoint.features)
    val featuresMatrix = new RowMatrix(features)
    val featuresMatrixSummary = featuresMatrix.computeColumnSummaryStatistics()

    val scaler = new StandardScaler(withMean = true, withStd = true).fit(features)
    val scaledDataRDDForAcceleration = dataRDDForAcceleration.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
    // compare the raw features with the scaled features
    //    println("dataRDDForAcceleration.first.features:" + dataRDDForAcceleration.first.features)
    //    println("scaledDataRDDForAcceleration.first.features:" + scaledDataRDDForAcceleration.first.features)

    val dataRDDForAccelerationWithIdx = scaledDataRDDForAcceleration.zipWithIndex().map(mapEntry => (mapEntry._2, mapEntry._1))

    val testDataRDDForAccelerationWithIdx = dataRDDForAccelerationWithIdx.sample(false, 0.1, 40)
    val trainDataRDDForAccelerationWithIdx = dataRDDForAccelerationWithIdx.subtract(testDataRDDForAccelerationWithIdx)

    val testDataRDDForAcceleration = testDataRDDForAccelerationWithIdx.map(mapEntry => mapEntry._2)
    val trainingDataRDDForAcceleration = trainDataRDDForAccelerationWithIdx.map(mapEntry => mapEntry._2)

    println("Training data size:" + trainingDataRDDForAcceleration.collect().size)
    println("Test data size:" + testDataRDDForAcceleration.collect().size)
    println()

    trainingDataRDDForAcceleration.cache

    val numIterations = 30
    val stepSize = 0.0001

    /*****  Decision Tree model for Acceleration start ****/
    val dTreeTrainedModelForAcceleration = DecisionTree.trainRegressor(trainingDataRDDForAcceleration, Map[Int, Int](), "variance", 5, 9)
    val dTreePredictedVsActualForAcceleration = testDataRDDForAcceleration.map { testDataRow =>
      (Math.round(dTreeTrainedModelForAcceleration.predict(testDataRow.features)).toDouble, testDataRow.label)
    }
    println("Predicted vs Actual value of Acceleration for Decision Tree model")
    dTreePredictedVsActualForAcceleration.collect().foreach(println)
    var metrics = new RegressionMetrics(dTreePredictedVsActualForAcceleration)
    println("Performance metrics for Decision Tree Model for Acceleration:")
    printMetrics(metrics)
    println()
    /*****  Decision Tree model for Acceleration end ****/

  }

  def printMetrics(regressionMetrics: RegressionMetrics): Unit = {
    println(s"MSE = ${regressionMetrics.meanSquaredError}")
    println(s"RMSE = ${regressionMetrics.rootMeanSquaredError}")
    println(s"R-squared = ${regressionMetrics.r2}")
    println(s"MAE = ${regressionMetrics.meanAbsoluteError}")
    println(s"Explained variance = ${regressionMetrics.explainedVariance}")
  }
}
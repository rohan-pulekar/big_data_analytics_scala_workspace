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
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.BoostingStrategy

object Problem2ForHorsePower {
  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local").setAppName("Assignment11-Problem2ForHorsePower")
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

    val manufacturerCategoriesRDD = cleanCarDataRDD.map(rowAsArrayOfValues => rowAsArrayOfValues(headersMap("Manufacturer"))).distinct().collect
    val originCategoriesRDD = cleanCarDataRDD.map(rowAsArrayOfValues => rowAsArrayOfValues(headersMap("Origin"))).distinct().collect

    val categoriesMap = manufacturerCategoriesRDD.union(originCategoriesRDD).zipWithIndex.toMap
    val numberOfCategories = categoriesMap.size
    println("Number of categories:" + numberOfCategories)

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

      val nonCategoryFeatures = rowAsArrayOfValues.slice(2, 3).union(rowAsArrayOfValues.slice(3, 4)).union(rowAsArrayOfValues.slice(7, 8)).map(feature => feature.toDouble)

      val features = categoryFeatures ++ nonCategoryFeatures

      LabeledPoint(label, Vectors.dense(features))
    }

    dataRDDForHorsePower.cache

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
    val stepSizeForLR = 0.0001

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

    /*****  Random Forest model for Horse Power start ****/
    val numTrees = 3
    val maxDepth = 4
    val maxBins = 9
    val randomForestTrainedModelForHorsePower = RandomForest.trainRegressor(trainingDataRDDForHorsePower, Map[Int, Int](),
      numTrees, "auto", "variance", maxDepth, maxBins)
    val randomForestPredictedVsActualForHorsePower = testDataRDDForHorsePower.map { testDataRow =>
      (Math.round(randomForestTrainedModelForHorsePower.predict(testDataRow.features)).toDouble, testDataRow.label)
    }
    println("Predicted vs Actual value of Horse Power for Random Forest model")
    randomForestPredictedVsActualForHorsePower.collect().foreach(println)
    metrics = new RegressionMetrics(randomForestPredictedVsActualForHorsePower)
    println("Performance metrics for Random Forest Model for Horse Power:")
    printMetrics(metrics)
    println()
    /*****  Random Forest model for Horse Power end ****/

    /*****  Random Forest model for Gradient Boosted Model start ****/
    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.numIterations = 10
    boostingStrategy.treeStrategy.maxDepth = 5
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    val gradientTrainedModelForHorsePower = GradientBoostedTrees.train(trainingDataRDDForHorsePower, boostingStrategy)
    val gradientPredictedVsActualForHorsePower = testDataRDDForHorsePower.map { testDataRow =>
      (Math.round(gradientTrainedModelForHorsePower.predict(testDataRow.features)).toDouble, testDataRow.label)
    }
    println("Predicted vs Actual value of Horse Power for Gradient Boosted model")
    gradientPredictedVsActualForHorsePower.collect().foreach(println)
    metrics = new RegressionMetrics(gradientPredictedVsActualForHorsePower)
    println("Performance metrics for Gradient Boosted Model for Horse Power:")
    printMetrics(metrics)
    println()
    /*****  Random Forest model for Gradient Boosted Model end ****/

    // I was unable to use SVMWithSGD and LogisticRegressionWithSGD since label is not a boolean (0 or 1)
    // I was unable to use naive bayes after scaled data because of negative values

  }

  def printMetrics(regressionMetrics: RegressionMetrics): Unit = {
    println(s"MSE = ${regressionMetrics.meanSquaredError}")
    println(s"RMSE = ${regressionMetrics.rootMeanSquaredError}")
    println(s"R-squared = ${regressionMetrics.r2}")
    println(s"MAE = ${regressionMetrics.meanAbsoluteError}")
    println(s"Explained variance = ${regressionMetrics.explainedVariance}")
  }
}
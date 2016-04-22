package e62.ml.assignment11

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
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

object Problem1 {
  def main(args: Array[String]) {

    val conf = new SparkConf().setMaster("local").setAppName("Assignment11")
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

    val entireDataRDD = cleanCarDataRDD.map { rowAsArrayOfValues =>
      var label = 0
      if (rowAsArrayOfValues(headersMap("Horsepower")) != "NaN") {
        label = rowAsArrayOfValues(headersMap("Horsepower")).toInt
      }
      var feature = 0.0
      if (rowAsArrayOfValues(headersMap("Displacement")) != "NaN") {
        feature = rowAsArrayOfValues(headersMap("Displacement")).toDouble
      }
      LabeledPoint(label, Vectors.dense(Array(feature)))
    }

    entireDataRDD.cache   

    val entireDataRDDWithIdx = entireDataRDD.zipWithIndex().map(mapEntry => (mapEntry._2, mapEntry._1))

    val testDataRDDWithIdx = entireDataRDDWithIdx.sample(false, 0.1, 40)
    val trainDataRDDWithIdx = entireDataRDDWithIdx.subtract(testDataRDDWithIdx)

    val testDataRDD = testDataRDDWithIdx.map(mapEntry => mapEntry._2)
    val trainingDataRDD = trainDataRDDWithIdx.map(mapEntry => mapEntry._2)
        
    println("Test data size:" + testDataRDD.collect().size)
    println("Training data size:" + trainingDataRDD.collect().size)
    println()

    trainingDataRDD.cache

    val numIterationsForLR = 100000
    val stepSizeForLR = 0.001

    /*****  Linear Regression with SGD model start ****/
    val linearWithSGDTrainedModel = LinearRegressionWithSGD.train(trainingDataRDD, numIterationsForLR, stepSizeForLR)
    val linearWithSGDPredictedVsActual = testDataRDD.map { testDataRow =>
      (Math.round(linearWithSGDTrainedModel.predict(testDataRow.features)).toDouble, testDataRow.label)
    }
    println("Predicted vs Actual value of Horse Power for Linear Regression with SGD")
    linearWithSGDPredictedVsActual.collect().foreach(println)
    var metrics = new RegressionMetrics(linearWithSGDPredictedVsActual)
    println("Performance metrics for Linear Regression:")
    printMetrics(metrics)
    println()
    val linearWithSGDOutputRDD = testDataRDD.map(testDataRow =>
      (testDataRow.features(0), Math.round(linearWithSGDTrainedModel.predict(testDataRow.features)).toDouble, testDataRow.label)).saveAsTextFile("./output_files/linearWithSGDOutput.txt")
    /*****  Linear Regression with SGD model end ****/

    /*****  Random Forest model start ****/
    val numTrees = 3
    val maxDepth = 4
    val maxBins = 9
    val randomForestTrainedModel = RandomForest.trainRegressor(trainingDataRDD, Map[Int, Int](),
      numTrees, "auto", "variance", maxDepth, maxBins)
    val randomForestPredictedVsActual = testDataRDD.map { testDataRow =>
      (Math.round(randomForestTrainedModel.predict(testDataRow.features)).toDouble, testDataRow.label)
    }
    metrics = new RegressionMetrics(randomForestPredictedVsActual)
    println("Performance metrics for Random Forest:")
    printMetrics(metrics)
    println()
    /*****  Random Forest model end ****/

    /*****  Gradient BoostedTrees model start ****/
    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.numIterations = 10
    boostingStrategy.treeStrategy.maxDepth = 5
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    val gradientTrainedModel = GradientBoostedTrees.train(trainingDataRDD, boostingStrategy)
    val gradientPredictedVsActual = testDataRDD.map { testDataRow =>
      (Math.round(gradientTrainedModel.predict(testDataRow.features)).toDouble, testDataRow.label)
    }
    metrics = new RegressionMetrics(gradientPredictedVsActual)
    println("Performance metrics for Gradient Boosted:")
    printMetrics(metrics)
    println()
    /*****  Gradient BoostedTrees model end ****/
    
    /*****  Naive Bayes model start ****/
    val naiveTrainedModelForHorsePower = NaiveBayes.train(trainingDataRDD)
    val naivePredictedVsActualForHorsePower = testDataRDD.map { testDataRow =>
      (Math.round(naiveTrainedModelForHorsePower.predict(testDataRow.features)).toDouble, testDataRow.label)
    }
    println("Predicted vs Actual value for Naive Bayes model")
    naivePredictedVsActualForHorsePower.collect().foreach(println)
    metrics = new RegressionMetrics(naivePredictedVsActualForHorsePower)
    println("Performance metrics for Naive Bayes Model for Horse Power:")
    printMetrics(metrics)
    println()
    /*****  Naive Bayes model end ****/

    // I was unable to use SVMWithSGD and LogisticRegressionWithSGD since label is not a boolean (0 or 1)
    // If I use standardscaler and scale the features linear regression model gives bad results.  so i have kept not scaled the features

  }

  def printMetrics(regressionMetrics: RegressionMetrics): Unit = {
    println(s"MSE = ${regressionMetrics.meanSquaredError}")
    println(s"RMSE = ${regressionMetrics.rootMeanSquaredError}")
    println(s"R-squared = ${regressionMetrics.r2}")
    println(s"MAE = ${regressionMetrics.meanAbsoluteError}")
    println(s"Explained variance = ${regressionMetrics.explainedVariance}")
  }
}
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

    val features = entireDataRDD.map(labeledPoint => labeledPoint.features)
    val featuresMatrix = new RowMatrix(features)
    val featuresMatrixSummary = featuresMatrix.computeColumnSummaryStatistics()

    val scaler = new StandardScaler(withMean = true, withStd = true).fit(features)
    val scaledEntireDataRDD = entireDataRDD.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
    // compare the raw features with the scaled features
    println("dataRDDForHorsePower.first.features:" + entireDataRDD.first.features)
    println("scaledDataRDDForHorsePower.first.features:" + scaledEntireDataRDD.first.features)

    val scaledEntireDataRDDWithIdx = scaledEntireDataRDD.zipWithIndex().map(mapEntry => (mapEntry._2, mapEntry._1))

    val testDataRDDWithIdx = scaledEntireDataRDDWithIdx.sample(false, 0.1, 40)
    val trainDataRDDWithIdx = scaledEntireDataRDDWithIdx.subtract(testDataRDDWithIdx)

    val testDataRDD = testDataRDDWithIdx.map(mapEntry => mapEntry._2)
    val trainingDataRDD = trainDataRDDWithIdx.map(mapEntry => mapEntry._2)

    println("Test data size:" + testDataRDD.collect().size)
    println("Training data size:" + trainingDataRDD.collect().size)
    println()

    trainingDataRDD.cache

    val numIterations = 20
    val step = 0.01

    /*****  Decision Tree model start ****/
    val decisionTreeTrainedModel = DecisionTree.trainRegressor(trainingDataRDD, Map[Int, Int](), "variance", 5, 9)
    val decisionTreePredictedVsActual = testDataRDD.map { testDataRow =>
      (decisionTreeTrainedModel.predict(testDataRow.features), testDataRow.label)
    }
    var metrics = new RegressionMetrics(decisionTreePredictedVsActual)
    println("Performance metrics for Decision Tree :")
    printMetrics(metrics)
    println()
    /*****  Decision Tree model end ****/

    /*****  Linear Regression with SGD model start ****/
    val linearWithSGDTrainedModel = LinearRegressionWithSGD.train(trainingDataRDD, numIterations, 0.1)
    val linearWithSGDPredictedVsActual = testDataRDD.map { testDataRow =>
      (linearWithSGDTrainedModel.predict(testDataRow.features), testDataRow.label)
    }
    println("Predicted vs Actual value of Horse Power for Linear Regression with SGD")
    linearWithSGDPredictedVsActual.collect().foreach(println)
    metrics = new RegressionMetrics(linearWithSGDPredictedVsActual)
    println("Performance metrics for Linear Regression:")
    printMetrics(metrics)
    println()
    //    val linearWithSGDOutputRDD = testDataRDD.map(testDataRow =>
    //      (testDataRow.features(0), linearWithSGDTrainedModel.predict(testDataRow.features), testDataRow.label)).saveAsTextFile("./output_files/linearWithSGDOutput.txt")
    /*****  Linear Regression with SGD model end ****/

    /*****  Random Forest model start ****/
    val numTrees = 3
    val maxDepth = 4
    val maxBins = 9
    val randomForestTrainedModel = RandomForest.trainRegressor(trainingDataRDD, Map[Int, Int](),
      numTrees, "auto", "variance", maxDepth, maxBins)
    val randomForestPredictedVsActual = testDataRDD.map { testDataRow =>
      (randomForestTrainedModel.predict(testDataRow.features), testDataRow.label)
    }
    /*****  Random Forest model end ****/

    /*****  Gradient BoostedTrees model start ****/
    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.numIterations = numIterations
    boostingStrategy.treeStrategy.maxDepth = 5
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    val gradientTrainedModel = GradientBoostedTrees.train(trainingDataRDD, boostingStrategy)
    val gradientPredictedVsActual = testDataRDD.map { testDataRow =>
      (gradientTrainedModel.predict(testDataRow.features), testDataRow.label)
    }
    metrics = new RegressionMetrics(gradientPredictedVsActual)
    println("Performance metrics for Gradient Boosted:")
    printMetrics(metrics)
    println()
    /*****  Gradient BoostedTrees model end ****/

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
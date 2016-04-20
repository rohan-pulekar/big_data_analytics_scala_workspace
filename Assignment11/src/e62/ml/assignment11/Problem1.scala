package e62.ml.assignment11

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.Vectors
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

    val rawCarDataRDDWithHeaders = sc.textFile("./data/Small_Car_Data_minimum.csv")

    val headersRow = rawCarDataRDDWithHeaders.first()

    println("headers:")
    println(headersRow)
    println()

    val uncleanCarDataRDDOfUnsplitLines = rawCarDataRDDWithHeaders.filter(rowAsAString => (rowAsAString != headersRow))

    val uncleanCarDataRDD = uncleanCarDataRDDOfUnsplitLines.map(rowAsAString => rowAsAString.split(","))

    val cleanCarDataRDD = uncleanCarDataRDD.map(rowAsArrayOfValues => rowAsArrayOfValues.map(value => value.trim()))

    val entireDataRDD = cleanCarDataRDD.map { rowAsArrayOfValues =>
      val label = rowAsArrayOfValues(4).toInt
      val feature = rowAsArrayOfValues(3).toDouble
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

    val numIterations = 20
    val step = 0.01

    val decisionTreeTrainedModel = DecisionTree.trainRegressor(trainingDataRDD, Map[Int, Int](), "variance", 5, 9)
    val decisionTreePredictedVsActual = testDataRDD.map { testDataRow =>
      (decisionTreeTrainedModel.predict(testDataRow.features), testDataRow.label)
    }
    var metrics = new RegressionMetrics(decisionTreePredictedVsActual)
    println("Performance metrics for Decision Tree :")
    printMetrics(metrics)
    println()
    
    val linearWithSGDTrainedModel = LinearRegressionWithSGD.train(trainingDataRDD, numIterations, 0.1)
    val linearWithSGDPredictedVsActual = testDataRDD.map { testDataRow =>
      (linearWithSGDTrainedModel.predict(testDataRow.features), testDataRow.label)
    }
    metrics = new RegressionMetrics(linearWithSGDPredictedVsActual)
    println("Performance metrics for Linear Regression:")
    printMetrics(metrics)
    println()
    val linearWithSGDOutputRDD = testDataRDD.map (testDataRow =>
      (testDataRow.features(0), linearWithSGDTrainedModel.predict(testDataRow.features), testDataRow.label)
    ).saveAsTextFile("./output_files/linearWithSGDOutput.txt")
    

    val naiveTrainedModel = NaiveBayes.train(trainingDataRDD)
    val naivePredictedVsActual = testDataRDD.map { testDataRow =>
      (naiveTrainedModel.predict(testDataRow.features), testDataRow.label)
    }
    metrics = new RegressionMetrics(naivePredictedVsActual)
    println("Performance metrics for Naive Bayes:")
    printMetrics(metrics)
    println()

    val numTrees = 3
    val maxDepth = 4
    val maxBins = 9
    val randomForestTrainedModel = RandomForest.trainRegressor(trainingDataRDD, Map[Int, Int](),
      numTrees, "auto", "variance", maxDepth, maxBins)
    val randomForestPredictedVsActual = testDataRDD.map { testDataRow =>
      (randomForestTrainedModel.predict(testDataRow.features), testDataRow.label)
    }
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

    // Unable to use SVMWithSGD

  }

  def printMetrics(regressionMetrics: RegressionMetrics): Unit = {
    println("bnm,")
    println(s"MSE = ${regressionMetrics.meanSquaredError}")
    println(s"RMSE = ${regressionMetrics.rootMeanSquaredError}")
    println(s"R-squared = ${regressionMetrics.r2}")
    println(s"MAE = ${regressionMetrics.meanAbsoluteError}")
    println(s"Explained variance = ${regressionMetrics.explainedVariance}")
  }
}
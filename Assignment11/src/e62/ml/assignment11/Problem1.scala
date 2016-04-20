package e62.ml.assignment11

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel

object Problem1 {
  def main(args: Array[String]) {

    val conf = new SparkConf().setMaster("local").setAppName("Assignment11")
    val sc = new SparkContext(conf)

    val rawCarDataRDDWithHeaders = sc.textFile("./data/Small_Car_Data_minimum.csv")

    val headersRow = rawCarDataRDDWithHeaders.first()

    println(headersRow)

    val uncleanCarDataRDDOfUnsplitLines = rawCarDataRDDWithHeaders.filter(rowAsAString => (rowAsAString != headersRow))
    uncleanCarDataRDDOfUnsplitLines.collect().foreach(println)

    val uncleanCarDataRDD = uncleanCarDataRDDOfUnsplitLines.map(rowAsAString => rowAsAString.split(","))

    val cleanCarDataRDD = uncleanCarDataRDD.map(rowAsArrayOfValues => rowAsArrayOfValues.map(value => value.trim()))

    val entireDataRDD = cleanCarDataRDD.map { rowAsArrayOfValues =>
      val label = rowAsArrayOfValues(4).toInt
      val feature = rowAsArrayOfValues(3).toDouble
      LabeledPoint(label, Vectors.dense(Array(feature)))
    }

    entireDataRDD.cache

    entireDataRDD.collect().foreach(println)

    entireDataRDD.zipWithIndex().collect().foreach(println)
    val entireDataRDDWithIdx = entireDataRDD.zipWithIndex().map(mapEntry => (mapEntry._2, mapEntry._1))

    entireDataRDDWithIdx.collect().foreach(println)

    val testDataRDDWithIdx = entireDataRDDWithIdx.sample(false, 0.1, 40)
    val trainDataRDDWithIdx = entireDataRDDWithIdx.subtract(testDataRDDWithIdx)

    val testDataRDD = testDataRDDWithIdx.map(mapEntry => mapEntry._2)
    val trainingDataRDD = trainDataRDDWithIdx.map(mapEntry => mapEntry._2)

    println(testDataRDD.collect().size)
    println(trainingDataRDD.collect().size)

    trainingDataRDD.cache

    val numIterations = 20
    val step = 0.01

    val decisionTreeTrainedModel = DecisionTree.trainRegressor(trainingDataRDD, Map[Int, Int](), "variance", 5, 9)
    val decisionTreePredictedVsActual = testDataRDD.map { testDataRow =>
      (decisionTreeTrainedModel.predict(testDataRow.features), testDataRow.label)
    }
    decisionTreePredictedVsActual.collect().foreach(println)

    val linearTrainedModel = LinearRegressionWithSGD.train(trainingDataRDD, numIterations, 0.1)
    val linearPredictedVsActual = testDataRDD.map { testDataRow =>
      (linearTrainedModel.predict(testDataRow.features), testDataRow.label)
    }
    linearPredictedVsActual.collect().foreach(println)

    val naiveTrainedModel = NaiveBayes.train(trainingDataRDD)
    val naivePredictedVsActual = testDataRDD.map { testDataRow =>
      (naiveTrainedModel.predict(testDataRow.features), testDataRow.label)
    }
    naivePredictedVsActual.collect().foreach(println)

    val numTrees = 3
    val maxDepth = 4
    val maxBins = 9
    val randomForestTrainedModel = RandomForest.trainRegressor(trainingDataRDD, Map[Int, Int](),
      numTrees, "auto", "variance", maxDepth, maxBins)
    val randomForestPredictedVsActual = testDataRDD.map { testDataRow =>
      (randomForestTrainedModel.predict(testDataRow.features), testDataRow.label)
    }
    randomForestPredictedVsActual.collect().foreach(println)
    println("------------")
    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.numIterations = numIterations
    boostingStrategy.treeStrategy.maxDepth = 5
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    val gradientTrainedModel = GradientBoostedTrees.train(trainingDataRDD, boostingStrategy)
    val gradientPredictedVsActual = testDataRDD.map { testDataRow =>
      (gradientTrainedModel.predict(testDataRow.features), testDataRow.label)
    }
    gradientPredictedVsActual.collect().foreach(println)

    // Unable to use SVMWithSGD

  }
}
package a.b

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object TestScalaClass {
    def main(args: Array[String]) {
      val sparkConf = new SparkConf().setMaster("local").setAppName("Test")
      val sparkContext = new SparkContext(sparkConf);
      val testRDD = sparkContext.parallelize(List ("Senator", "Marco", "Rubio", "10", "the", "Puerto", "Rico."))
      val filteredTestRDD = testRDD.filter(word => word.matches("[A-Za-z.]+")).map(word => word.replaceAll("[^A-Za-z0-9]", ""))
      filteredTestRDD.collect().foreach(println)
      
      "jhkl".split(" ")
    }
}
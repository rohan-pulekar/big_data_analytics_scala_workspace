package edu.hu.bigdata.e63ScalaSection

import scala.math.random
import org.apache.spark._

object SparkPi {

def main(args: Array[String]): Unit = {

  val conf = new SparkConf().setMaster("local").setAppName("SparkPi")
  val spark = new SparkContext(conf)
  spark.setCheckpointDir("/tmp/spark/")
  println("-------------Attach debugger now!--------------")
  Thread.sleep(8000)
    
  val slices = if (args.length > 0) args(0).toInt else 2
  val n = math.min(1000L * slices, Int.MaxValue).toInt // avoid overflow
  val count = spark.parallelize(1 until n, slices).map 
    { i =>
      val x = random * 2 - 1
      val y = random * 2 - 1
      if (x*x + y*y < 1) 1 
        else 0
      }.reduce(_ + _)

println("Pi is roughly " + 4.0 * count / n)
spark.stop()
}
}


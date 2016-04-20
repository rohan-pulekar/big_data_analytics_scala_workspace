package edu.hu.bigdata.e63ScalaSection

import org.apache.spark._
import org.apache.spark.SparkContext._

object PartionedWC {
  def main(args:Array[String]) :Unit ={
   
    val inputfile=args(0)
    val outfile=args(1)
   
    val conf = new SparkConf().setMaster("local").setAppName("partitionedWC")   
    val sc = new SparkContext(conf)
    val RDD5 = sc.textFile(inputfile,66)
    println("RDD5 Partions =" + RDD5.partitions.size)
    println("RDD5 Partions =" + RDD5.partitions.length)
    val RDD1= sc.textFile(inputfile)
     println("RDD1 Partions =" + RDD1.partitions.size)
    println("RDD1 Partions =" + RDD1.partitions.length)
    RDD5.take(2)
    RDD1.take(2)

  }

}
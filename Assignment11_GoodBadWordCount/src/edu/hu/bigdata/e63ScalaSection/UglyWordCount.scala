package edu.hu.bigdata.e63ScalaSection

/**
 * @author joglekarrb
 * This wordcount uses groupByKey
 */

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import java.io.File
import org.apache.commons.io._;

object UglyWordCount {
  def main(args: Array[String]): Unit={
    val inputFile = args(0)
    val outputFile = args(1)
    
   try {
    FileUtils.deleteDirectory (new File(outputFile))
        println ("Files Deleted ")
    } catch 
    {  
       case _: Throwable => println("Got some other kind of exception")
    }
     
    
    val timerstart = System.currentTimeMillis()
    val conf = new SparkConf().setMaster("local[*]").setAppName("wordCount")
    // Create a Scala Spark Context.
    val sc = new SparkContext(conf)
    // Load our input data.
    val mypartitions=4
    val inputRDD = sc.textFile(inputFile,mypartitions )
    val sze=inputRDD.partitions.size
    
    
    //Split up into words - Used groupByKey
    val words = inputRDD
                .flatMap(_.split(" "))
                .map(word=> (word, 1))
                .groupByKey()
                .map {case (w, counts) => (w, counts.sum)}
    words.saveAsTextFile(outputFile)
    val elapsedtime = System.currentTimeMillis() - timerstart
    println ("Size of InputRDD : " +  sze  + "partitions" )
    println("Execution Time Using GroupbyKey : " + elapsedtime/1000.0 + " sec" ) 
  }
}
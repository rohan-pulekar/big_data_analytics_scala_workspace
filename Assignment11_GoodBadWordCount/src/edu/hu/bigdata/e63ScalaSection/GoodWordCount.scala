/**
 *
 */
package edu.hu.bigdata.e63ScalaSection


/**
 * @author joglekarrb
 *
 */
import org.apache.spark._
import org.apache.spark.SparkContext._
import java.io.File
import org.apache.commons.io._;

object GoodWordCount {
  def main(args: Array[String]): Unit={
    val inputFile = args(0)
    val outputFile = args(1)
 
//Clean Output to make it rerunnable 
    
   try {
    FileUtils.deleteDirectory (new File(outputFile))
        println ("Files Deleted ")
    } catch 
    {  
       case _: Throwable => println("Something is really wrong")
    }
      
    // Timer Start recording 
    val timerstart = System.currentTimeMillis()
    // Timer Started 
    
    
    val conf = new SparkConf().setAppName("wordCount").setMaster("local[*]")
    // Create a Scala Spark Context.
    val sc = new SparkContext(conf)
    // Load our input data.
    val input = sc.textFile(inputFile)
    val sze=input.partitions.size
    // Split up into words.
    val words = input.flatMap(line => line.split(" "))
    // Transform into word and count.
    val counts = words.map(word => (word, 1)).reduceByKey { case (x, y) => x + y }
    // Save the word count back out to a text file, causing evaluation.
    counts.saveAsTextFile(outputFile)
    
    println ("Size of InputRDD : " +  sze  + "partitions")
       
    //Timer Stop
    val elapsedtime = System.currentTimeMillis() - timerstart
     println("Execution Time : " + elapsedtime/1000.0 + "s" )
    
  }
}
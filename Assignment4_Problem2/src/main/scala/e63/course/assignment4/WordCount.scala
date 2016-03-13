package e63.course.assignment4

import org.apache.spark._
import org.apache.spark.SparkContext._

// This program is for the Assignment4 Problem2 of e63 course (Big Data Analytics)
object WordCount {
  def main(args: Array[String]) {
    
      // set input and out files/dirs
      val inputFileName = args(0)
      val outputDirName = args(1)
      
      // Create a Spark Configuration
      val sparkConf = new SparkConf().setAppName("Assignment4_Problem2")
      
      // Create a Scala Spark Context.
      val sparkContext = new SparkContext(sparkConf)
      
      // Load the input data.
      val inputFileContents =  sparkContext.textFile(inputFileName)
      
      // Split the input data into words.
      val wordsListRDD = inputFileContents.flatMap(line => line.split(" "))
      
      // Transform into a map of word and default count of 1.
      // The below replaceAll() is called to eliminate punctuations and to
			// convert the words into lower case
      val wordsListWithDefaultCount = wordsListRDD.map(word => (word.replaceAll("[^A-Za-z0-9]", "").toLowerCase(), 1))
      
      // Filter the list to eliminate blanks
      val filteredWordsList = wordsListWithDefaultCount.filter{case (word, count) => !word.isEmpty()}
      
      // Call reduceByKey to count occurrence of each word
      val wordsAndTheirCount = filteredWordsList.reduceByKey{case (x, y) => x + y}
      
      // Sort the list by word (which is the key)
      val sortedWordsAndTheirCount = wordsAndTheirCount.sortByKey()
      
      // Save the word count back out to a text file, causing evaluation.
      sortedWordsAndTheirCount.saveAsTextFile(outputDirName)
    }
}
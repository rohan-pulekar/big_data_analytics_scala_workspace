package e63.course.assignment6

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

/**
 * @author Rohan Pulekar
 * Purpose of this program:  
 * This program is for Assignment6 Problem1 of e63 course (Big Data Analytics) of Spring 2016 batch of Harvard Extension School
 * 
 */
object Problem1 {
  def main(args: Array[String]) {
        
    // set input file paths
    val paragraphAFileName = args(0)
    val paragraphBFileName = args(1)
          
    // Create a Spark Configuration
    val sparkConf = new SparkConf().setAppName("Assignment6_Problem1")
      
    // Create a Scala Spark Context.
    val sparkContext = new SparkContext(sparkConf);
    
    // Load the input data into RDDs
    val paragraphA = sparkContext.textFile(paragraphAFileName)
    val paragraphB = sparkContext.textFile(paragraphBFileName)
    
    // get an RDD of tokens for each of the input files
    val paragraphATokens = paragraphA.flatMap(line => line.split(" "))
    val paragraphBTokens = paragraphB.flatMap(line => line.split(" "))
    
    // print out number of tokens
    println("\nTotal number of tokens in paragraphA: " + paragraphATokens.count());
    println("\nTotal number of tokens in paragraphB: " + paragraphBTokens.count());
    
    // To transform the RDDs into RDDs that contain only words
    // filter in only the words from the tokens
    // '.' and ',' are treated separately because those characters immediately follow a word, in which case we want to keep the word and get rid of '.' and ','
    val paragraphAWords = paragraphATokens.filter(word => word.matches("[A-Za-z0-9.,]+")).map(word => word.toLowerCase().replaceAll("[.,]", ""))
    val paragraphBWords = paragraphBTokens.filter(word => word.matches("[A-Za-z0-9.,]+")).map(word => word.toLowerCase().replaceAll("[.,]", ""))
    
    // list first 10 words from paragraphA and paragrahB
    println("\nFirst 10 words from paragraphA:")
    paragraphAWords.take(10).foreach(println)
    println("\nFirst 10 words from paragraphB:")
    paragraphBWords.take(10).foreach(println)
    
    // print out number of words
    println("\nTotal number of words in paragraphA: " + paragraphAWords.count());
    println("\nTotal number of words in paragraphB: " + paragraphBWords.count());
    
    // create RDDs that contain only unique words from each RDD
    val paragraphAUniqueWords = paragraphAWords.distinct()
    val paragraphBUniqueWords = paragraphBWords.distinct()
    
    // print out number of unique words
    println("\nTotal number of unique words in paragraphA: " + paragraphAUniqueWords.count());
    println("\nTotal number of unique words in paragraphB: " + paragraphBUniqueWords.count());
     
    // create an RDD with words that are in paragraphA but not in paragraphB
    val wordsInParAButNotInParB = paragraphAUniqueWords.subtract(paragraphBUniqueWords)
    
    // print out information of words that are in paragraphA but not in paragraphB
    println("\nTotal number of words in paragraphA that are not in paragraphB: " + wordsInParAButNotInParB.count())
    println("\nList of words in paragraphA that are not in paragraphB:")
    wordsInParAButNotInParB.collect().foreach(println)
    
    // create an RDD with words that are common in paragraphA and paragraphB
    val wordsCommonInParAAndParB = paragraphAUniqueWords.intersection(paragraphBUniqueWords)
          
    // print out information of words that are common in paragraphA and pragraphB
    println("\nTotal number of common words in paragraphA and paragraphB: " + wordsCommonInParAAndParB.count())
    println("\nList of words that are common in paragraphA and paragraphB")
    wordsCommonInParAAndParB.collect().foreach(println)
        
  }
}
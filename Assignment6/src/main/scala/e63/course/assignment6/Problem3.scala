package e63.course.assignment6

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.types.LongType
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.FloatType

/**
 * @author Rohan Pulekar
 * Purpose of this program:  
 * This program is for Assignment6 Problem3 of e63 course (Big Data Analytics) of Spring 2016 batch of Harvard Extension School
 * 
 */
object Problem3 {
  def main(args: Array[String]) {

    // set input files
    val ebayFilePath = args(0)

    // Create a Spark Configuration
    val sparkConf = new SparkConf().setAppName("Assignment6_Problem3")

    // Create a Scala Spark Context.
    val sparkContext = new SparkContext(sparkConf);

    // create a sql context
    val sqlContext = new SQLContext(sparkContext)

    // import data into an RDD
    val ebayRDD = sparkContext.textFile(ebayFilePath)

    // transform the RDD into another RDD made of tuples
    val ebayRDDOfTuples = ebayRDD.map(l => new Tuple9(l.split(",")(0), l.split(",")(1), l.split(",")(2), l.split(",")(3), l.split(",")(4), l.split(",")(5), l.split(",")(6), l.split(",")(7), l.split(",")(8)))

    // transform the RDD into RDD of Rows
    val ebayRDDOfRows = ebayRDDOfTuples.map(e => Row(e._1.toLong, e._2.toFloat, e._3.toFloat, e._4, e._5.toInt, e._6.toFloat, e._7.toFloat, e._8, e._9.toInt))

    // print first 10 rows to make sure RDDs have been generated correctly
    println("ebayRDDOfRows:")
    ebayRDDOfRows.take(10).foreach(println)

    // create schema for ebay.csv file
    val ebaySchema = StructType(StructField("auctionid", LongType, false) ::
      StructField("bid", FloatType, false) ::
      StructField("bidtime", FloatType, false) ::
      StructField("bidder", StringType, false) ::
      StructField("bidderrate", IntegerType, false) ::
      StructField("openbid", FloatType, false) ::
      StructField("price", FloatType, false) ::
      StructField("item", StringType, false) ::
      StructField("daystolive", IntegerType, false) :: Nil)

    // transform the RDD into dataframe with the help of above created schema
    val Auction = sqlContext.createDataFrame(ebayRDDOfRows, ebaySchema)

    // print schema of Auction dataframe
    println("Printing Auction dataframe schema")
    Auction.printSchema()

    // print first 10 rows of Auction data frame to make sure it is generated correctly
    println("Printing first 10 rows of Auction data frame to make sure data frame is created correctly")
    Auction.select("auctionid", "bid", "bidtime", "bidder", "bidderrate", "openbid", "price", "item", "daystolive").show(10)

    // how many auctions were held?  (querying the dataframe)
    println("Below is answer to the question:  How many auctions were held? (querying the dataframe)")
    println(Auction.select("auctionid").distinct().count())

    // How many bids were made per item? (querying the dataframe)
    println("Below is answer to the question: 	How many bids were made per item? (querying the dataframe)")
    Auction.groupBy("item").count().show()

    // What's the minimum bid (price) per item? (querying the dataframe)
    println("Below is answer to the question: What's the minimum bid (price) per item? (querying the dataframe)")
    Auction.groupBy("item").min("bid").show()

    // What's the maximum bid (price) per item? (querying the dataframe)
    println("Below is answer to the question: What's the maximum bid (price) per item? (querying the dataframe)")
    Auction.groupBy("item").max("bid").show()

    // What's the average bid (price) per item? (querying the dataframe)
    println("Below is answer to the question: What's the average bid (price) per item? (querying the dataframe)")
    Auction.groupBy("item").avg("bid").show()

    // create a data frame of item, auctionid and count of bids for that combination of item-auctionid
    val AuctionBidCountDataFrame = Auction.groupBy("item", "auctionid").count()

    // What is the minimum number of bids per item? (querying the dataframe)
    // Here I have shown min number of bids per item across the auctions
    // So first I have grouped by item and auctionid on line 93.  
    // Then I found the min number of bids placed in an auction for each item 
    println("Below is answer to the question: What is the minimum number of bids per item? (querying the dataframe)")
    AuctionBidCountDataFrame.groupBy("item").min("count").show()

    // What is the maximum number of bids per item? (querying the dataframe)
    // Here I have shown max number of bids per item across the auctions
    // So first I have grouped by item and auctionid on line 93.  
    // Then I found the max number of bids placed in an auction for each item 
    println("Below is answer to the question: What is the maximum number of bids per item? (querying the dataframe)")
    AuctionBidCountDataFrame.groupBy("item").max("count").show()

    // What is the average number of bids per item? (querying the dataframe)
    // Here I have shown avg number of bids per item across the auctions
    // So first I have grouped by item and auctionid on line 93.  
    // Then I found the avg number of bids placed in an auction for each item 
    println("Below is answer to the question: What is the average number of bids per item? (querying the dataframe)")
    AuctionBidCountDataFrame.groupBy("item").avg("count").show()

    // Show the bids with price > 100  (querying the dataframe)
    // Assumption: 'price' in the problem statement refers to closing price
    // If it meant bid price I would have used   .filter("bid>100")   instead
    println("Below is answer to the question: Show the bids with price > 100  (querying the dataframe)")
    Auction.filter("price > 100").show()

    // convert auction dataframe into a temporary table
    Auction.registerTempTable("Auction_temp_table")

    // How many auctions were held? (querying the temp sql table)
    println("Below is answer to the question: How many auctions were held? (querying the temp sql table)")
    sqlContext.sql("select count(distinct auctionid) from Auction_temp_table").show()

    // How many bids were made per item? (querying the temp sql table)
    println("Below is answer to the question: How many bids were made per item? (querying the temp sql table)")
    sqlContext.sql("select item, count(*) as number_of_bids from Auction_temp_table group by item").show()

    // What's the minimum bid (price) per item? (querying the temp sql table)
    println("Below is answer to the question: What's the minimum bid (price) per item? (querying the temp sql table)")
    sqlContext.sql("select item, min(bid) as minimum_bid_price from Auction_temp_table group by item").show()

    // What's the maximum bid (price) per item? (querying the temp sql table)
    println("Below is answer to the question: What's the maximum bid (price) per item? (querying the temp sql table)")
    sqlContext.sql("select item, max(bid) as maximum_bid_price from Auction_temp_table group by item").show()

    // What's the average bid (price) per item? (querying the temp sql table)
    println("Below is answer to the question: What's the average bid (price) per item? (querying the temp sql table)")
    sqlContext.sql("select item, avg(bid) as average_bid_price from Auction_temp_table group by item").show()

    // What is the minimum number of bids per item? (querying the temp sql table)
    // Here I have shown min number of bids per item across the auctions
    // So first I have grouped by item and auctionid
    // and then I found the min number of bids placed in an auction for each item 
    println("Below is answer to the question: What is the minimum number of bids per item? (querying the temp sql table)")
    sqlContext.sql("select item, min(number_of_bids) as min_number_of_bids from (select item, auctionid, count(*) as number_of_bids from Auction_temp_table group by item, auctionid order by item) auction_bid_count group by item").show()
    
    // What is the maximum number of bids per item? (querying the temp sql table)
    // Here I have shown max number of bids per item across the auctions
    // So first I have grouped by item and auctionid
    // and then I found the max number of bids placed in an auction for each item 
    println("Below is answer to the question: What is the maximum number of bids per item? (querying the temp sql table)")
    sqlContext.sql("select item, max(number_of_bids) as max_number_of_bids from (select item, auctionid, count(*) as number_of_bids from Auction_temp_table group by item, auctionid order by item) auction_bid_count group by item").show()

    // What is the average number of bids per item? (querying the temp sql table)
    // Here I have shown avg number of bids per item across the auctions
    // So first I have grouped by item and auctionid
    // and then I found the avg number of bids placed in an auction for each item 
    println("Below is answer to the question: What is the average number of bids per item? (querying the temp sql table)")
    sqlContext.sql("select item, avg(number_of_bids) as avg_number_of_bids from (select item, auctionid, count(*) as number_of_bids from Auction_temp_table group by item, auctionid order by item) auction_bid_count group by item").show()

    // Show the bids with price > 100  (querying the temp sql table)
    // Assumption: 'price' in the problem statement refers to closing price
    // If it meant bid price I would have used   ...where bid>100")   instead
    println("Below is answer to the question: Show the bids with price > 100  (querying the temp sql table)")
    sqlContext.sql("select * from Auction_temp_table where price>100").show()

  }
}
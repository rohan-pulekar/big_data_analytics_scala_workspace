package e63.course.assignment6

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import scala.Tuple3
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext
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
 * This program is for Assignment6 Problem2 of e63 course (Big Data Analytics) of Spring 2016 batch of Harvard Extension School
 * 
 */
object Problem2 {
  def main(args: Array[String]) {
        
    // set input files
    val empsFilePath = args(0)
    
    // Create a Spark Configuration
    val sparkConf = new SparkConf().setAppName("Assignment6_Problem2")
      
    // Create a Scala Spark Context.
    val sparkContext = new SparkContext(sparkConf);
    
    // create a sql context
    val sqlContext = new SQLContext(sparkContext)
    
    // create RDD emps of the input text file where every record in RDD will be a line in the txt file
    val emps = sparkContext.textFile(empsFilePath)
    
    // list out contents of emps RDD
    println("\n emps RDD contents:")
    emps.take(10).foreach(println)
    
    // create RDD emps_fields where every record will be a tuple of 3 elements
    val emps_fields = emps.map(line => new Tuple3((line.split(", ")(0)), (line.split(", ")(1)), (line.split(", ")(2))))

    // list out contents of emps_fields RDD
    println("\n emps_fields RDD contents:")
    emps_fields.take(10).foreach(println)    
    
    // create employees RDD where every record will be a Row
    val employees = emps_fields.map(e => Row(e._1, e._2.toInt, e._3.toFloat))
    
    // list out contents of employees RDD
    println("\n employees RDD contents")
    employees.take(10).foreach(println)
    
    // create a schema for the input txt file
    val empsSchema = StructType(StructField("name", StringType, false) ::
     StructField("age", IntegerType, false) ::
     StructField("salary", FloatType, false) :: Nil)
    
     // create a dataframe from the employees RDD and schema
    val employeesDataFrame = sqlContext.createDataFrame(employees, empsSchema)
    
    // print the content of employees data frame
    println("\nContent of employees data frame:")
    employeesDataFrame.select("name", "age", "salary").show()
    
    // transform the data frame into temporary table
    employeesDataFrame.registerTempTable("employees_temp_table")
    
    // select names of employees who have salary greater than 3500 from employees data frame
    println("Will now select from employeesDataFrame, names of employees who have salary greater than 3500")
    employeesDataFrame.filter("salary>3500").select("name").show()
    
    // select names of employees who have salary greater than 3500 from employees temporary table
    println("Will now select from employees_temp_table, names of employees who have salary greater than 3500")
    sqlContext.sql("select name from employees_temp_table where salary>3500").show()
  }
}
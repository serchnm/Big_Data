// Start a Spark Session
import org.apache.spark.sql.SparkSession

// Optional: Use the following code below to set the Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Spark Session
val spark = SparkSession.builder().getOrCreate()

// Import clustering Algorithm
import org.apache.spark.ml.clustering.KMeans

// Loads data.
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Wholesale_customers_data.csv")
data.show()
data.printSchema()
// val dataset = spark.read.option("header","true").option("inferSchema","true").csv("sample_kmeans_data.txt")
val feature_data = data.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper",$"Delicassen")
feature_data.show()
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val Assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")
val Traning= Assembler.transform(feature_data)
// Trains a k-means model.
val kmeans = new KMeans().setK(3).setSeed(12345L)
val model = kmeans.fit(Traning)

// Evaluate clustering by computing Within Set Sum of Squared Errors.
val WSSSE = model.computeCost(Traning)
println(s"Within Set Sum of Squared Errors = $WSSSE")

// Shows the result.
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
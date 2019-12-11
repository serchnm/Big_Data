import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

import org.apache.spark.ml.clustering.KMeans

val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("wholesale_customers_data.csv")

val assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")

val output = assembler.transform(data)
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(output)

// Evaluate clustering by calculate Within Set Sum of Squared Errors.
val WSSE = model.computeCost(output)
println(s"Within set sum of Squared Errors = $WSSE")

// Show results
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.SparkSession

// Se crea una sesion de spark
val spark = SparkSession.builder.appName("SummarizerExample").getOrCreate()
// Importaciones
import spark.implicits._
import Summarizer._

// Se crea una secuencia de datos a partir de vectores
val data = Seq((Vectors.dense(2.0, 3.0, 5.0), 1.0), (Vectors.dense(4.0, 6.0, 7.0), 2.0))
// Se crea un dataFrame con etiquetas definidas
val df = data.toDF("features", "weight")
// Se crea una tupla con los valores de las etiqeutas
val (meanVal, varianceVal) = df.select(metrics("mean", "variance").summary($"features",$"weight").as("summary")).select("summary.mean", "summary.variance").as[(Vector, Vector)].first()
// se imprimen los resultados por varianza y media
println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")
// Se imprimen los resultados por caracteristicas
val (meanVal2, varianceVal2) = df.select(mean($"features"), variance($"features")).as[(Vector, Vector)].first()

// Se imprimen los resultados sin la media y la suma
println(s"without weight: mean = ${meanVal2}, sum = ${varianceVal2}")

spark.stop()
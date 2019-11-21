import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row

// Se genera una secuencia de datos desde vectores
val data = Seq(Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),Vectors.dense(4.0, 5.0, 0.0, 3.0),Vectors.dense(6.0, 7.0, 0.0, 8.0),Vectors.sparse(4, Seq((0, 9.0), (3, 1.0))))

// Se transforman los datos y se genera un dataframe
val df = data.map(Tuple1.apply).toDF("features")
// Se toma una fila y se muestra la correlacion 
val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
println(s"Pearson correlation matrix:\n $coeff1")
// Se toma una fila y se muestra la correlacion 
val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
println(s"Spearman correlation matrix:\n $coeff2")
//Importamos las librerias necesarias con las que vamos a trabajar
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.log4j._
//Quita los warnings
Logger.getLogger("org").setLevel(Level.ERROR)

//Creamos una sesion de spark y cargamos los datos del CSV en un datraframe
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
//Desblegamos los tipos de datos.
df.printSchema()
df.show(1)

//Cambiamos la columna y por una con datos binarios.
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
//Desplegamos la nueva columna
newcolumn.show(1)

//Generamos la tabla features
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
//Mostramos la nueva columna
fea.show(1)
//Cambiamos la columna y a la columna label
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1)

//Logistic Regresion
val logistic = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
// Fit del modelo
val logisticModel = logistic.fit(feat)
//Impresion de los coegicientes y de la intercepcion
println(s"Coefficients: ${logisticModel.coefficients} Intercept: ${logisticModel.intercept}")
val logisticMult = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
val logisticMultModel = logisticMult.fit(feat)
println(s"Multinomial coefficients: ${logisticMultModel.coefficientMatrix}")
println(s"Multinomial intercepts: ${logisticMultModel.interceptVector}")
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

//Se imporortan lod datos del data set Bank
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")
// se hace el proceso de categorizar las variables tipo string a numerico
val yes = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val clean = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val cleanData = clean.withColumn("y",'y.cast("Int"))

//Se crea un array de los datos seleccionados
val featureCols = Array("age","previous","balance","duration")
//Se crea el vector con las columna deatures
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
//Se transforma en un nuevo df
val df2 = assembler.transform(cleanData)
//Se da un nuevo nobre a la columna y a label
val featuresLabel = df2.withColumnRenamed("y", "label")
//Selecciona los indices 
val dataI = featuresLabel.select("label","features")
//Se crea un array con los datos de entrenamiento y de test
val Array(training, test) = dataI.randomSplit(Array(0.7, 0.3), seed = 12345)
//Se declara el modelo de regresion
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

val lrModel = lr.fit(training)

println(s"Coefficients: \n${lrModel.coefficientMatrix}")
println(s"Intercepts: \n${lrModel.interceptVector}")

val trainingSummary = lrModel.summary
val accuracy = trainingSummary.accuracy
val falsePositiveRate = trainingSummary.weightedFalsePositiveRate
val truePositiveRate = trainingSummary.weightedTruePositiveRate
val fMeasure = trainingSummary.weightedFMeasure
val precision = trainingSummary.weightedPrecision
val recall = trainingSummary.weightedRecall

println(s"Accuracy: $accuracy\nFPR: $falsePositiveRate\nTPR: $truePositiveRate\n" +
  s"F-measure: $fMeasure\nPrecision: $precision\nRecall: $recall")

// Para medir el rendimiento 
val runtime = Runtime.getRuntime
val startTimeMillis = System.currentTimeMillis()

val mb = 0.000001
println("Used Memory: " + (runtime.totalMemory - runtime.freeMemory) * mb)
println("Free Memory: " + runtime.freeMemory * mb)
println("Total Memory: " + runtime.totalMemory * mb)
println("Max Memory: " + runtime.maxMemory * mb)


val endTimeMillis = System.currentTimeMillis()
val durationSeconds = (endTimeMillis - startTimeMillis) / 1000
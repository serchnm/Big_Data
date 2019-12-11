import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

//Se imporortan lod datos del data set Bank
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")
// se hace el proceso de categorizar las variables tipo string a numerico
val yes = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val no = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val cleanData = no.withColumn("y",'y.cast("Int"))

//Se crea un array de los datos seleccionados
val featureCols = Array("balance","day","duration","pdays","previous")
//Se crea el vector con las columna deatures
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
//Se transforma en un nuevo df
val features = assembler.transform(cleanData)
//Se cabia el nombre de la columna y a label
val featuresLabel = features.withColumnRenamed("y", "label")
//se indexan la columna label y features
val dataIndexed = featuresLabel.select("label","features")
//se dividen los datos de entrenamiento y test
val splits = dataIndexed.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

//Se crea la configuracion de las capas 
val layers = Array[Int](5,2,2,4)

//se declara el algoritmo
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
//se entrena al modelo
val model = trainer.fit(train)


val result = model.transform(test)

// Se selecciona la prediccion y la etiqueta
val predictionAndLabels = result.select("prediction", "label")

//Se evalua la precision del modelo
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
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
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

//Se imporortan lod datos del data set Bank
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")
// se hace el proceso de categorizar las variables tipo string a numerico
val yes = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val clean = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))

//Se crea un array de los datos seleccionados
val featureCols = Array("age","previous","balance","duration")
//Se crea el vector con las columna deatures
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
//Se transforma en un nuevo df
val df2 = assembler.transform(data)
//Se da un nuevo nobre a la columna y a label
val featuresLabel = df2.withColumnRenamed("y", "label")
//Selecciona los indices 
val dataIndexed = featuresLabel.select("label","features")
//Se indexan las columnas
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed)
//Se crea un array con los datos de entrenamiento y de test
val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))
//Se declara el abrbol de clasificacion que contendra los indices
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
//Se entrena
val model = pipeline.fit(trainingData)
//Se hace la prediccion de datos
val predictions = model.transform(testData)

//Imprime las predicciones 
predictions.select("predictedLabel", "label", "features").show(5)
//Se evalua la presicion de la prediccion 
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
// Para medir el rendimiento 
val runtime = Runtime.getRuntime
val startTimeMillis = System.currentTimeMillis()

val mb = 0.000001
println("Used Memory: " + (runtime.totalMemory - runtime.freeMemory) * mb)
println("Free Memory: " + runtime.freeMemory * mb)
println("Total Memory: " + runtime.totalMemory * mb)
println("Max Memory: " + runtime.maxMemory * mb)


val endTimeMillis = System.currentTimeMillis()
val durationSeconds = (startTimeMillis / 1000)
println("Duration:" + startTimeMillis / 1000)


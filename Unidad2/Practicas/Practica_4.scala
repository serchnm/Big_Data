import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Cargue y analice el archivo de datos, convirtiéndolo en un DataFrame.
val data = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")

// Índice de etiquetas, agregando metadatos a la columna de etiquetas.
// Se ajusta a todo el conjunto de datos para incluir todas las etiquetas en el índice.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
// Identifica automáticamente las características categóricas e indízalas.
// Establezca maxCategories para que las entidades con> 4 valores distintos se traten como continuas.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// Divide los datos en conjuntos de entrenamiento y prueba (30% para pruebas).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Entrena un modelo GBT.
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")

// Convierte las etiquetas indexadas de nuevo a etiquetas originales.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Indizadores de cadena y GBT en una canalización.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// Modelo de tren. Esto también ejecuta los indexadores.
val model = pipeline.fit(trainingData)

// Se hacen las predicciones
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Seleccione (predicción, etiqueta verdadera) y calcule el error de prueba.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1.0 - accuracy}")

// se genera el modelo de GBT
val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")
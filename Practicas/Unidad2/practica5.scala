import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Cargue los datos almacenados en formato LIBSVM como un DataFrame.
val data = spark.read.format("libsvm").load("data/sample_multiclass_classification_data.txt")

// Divide los datos en tren y prueba
val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

// especifica capas para la red neuronal:
// capa de entrada de tamaño 4 (características), dos intermedios de tamaño 5 y 4
// y salida de tamaño 3 (clases)
val layers = Array[Int](4, 5, 4, 3)

// crea el entrenador y establece sus parámetros
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// entrenar al modelo
val model = trainer.fit(train)

// calcular la precisión en el conjunto de prueba
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
// Se evalua la precision de las predicciones 
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
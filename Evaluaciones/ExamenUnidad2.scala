// Import the libraries to use
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.feature.StringIndexer
// Load the data stored in LIBSVM format as a DataFrame.
val data = spark.read.format("csv").option("header", "true").load("iris.csv")
val df2 = data.withColumn("sepal_length", data("sepal_length").cast(IntegerType))
val df3 = df2.withColumn("sepal_width", data("sepal_width").cast(IntegerType))
val df4 = df3.withColumn("petal_length", data("petal_length").cast(IntegerType))
val df5 = df4.withColumn("petal_width", data("petal_width").cast(IntegerType))
val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).setOutputCol("features"))
val output = assembler.transform(df5)

val indexer = new StringIndexer().setInputCol("species").setOutputCol("label")
val indexed = indexer.fit(output).transform(output)
// Split the data into train and test
val splits = indexed.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

// specify layers for the neural network:
// input feature layers of size |4| , two intermediate of size |5| and |4| and output of size |3| (classes)
val layers = Array[Int](4, 5, 4, 3)

// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// train the model
val model = trainer.fit(train)

// compute accuracy on the test set
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")


println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

//d.-
//sigmoidal Function:
// Se trata de una funcion continua no lineal que posee un rango 
//comprendido entre 0 y 1 aplicando un proceso a las unidades que sea
// cual sea el dato de entrada y salida simepre estara entre esos dos parametros

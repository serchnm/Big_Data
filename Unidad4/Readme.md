## Theoretical framework
#### Decision Tree
* [Decisions Tree](#Decisions-Tree) | [Characteristics-of-a-decision-tree](#Characteristics-of-a-decision-tree) | [Advantage](#Advantage) | [Disadvantages](#Disadvantages) | [Code](#Code1)

#### Logistic Regression
* [Logistic-Regresion](#Logistic-Regresion) | [Logistic-Regression-types](#Logistic-Regression-types) | [Decision-limit](#Decision-limit) | [Code](#Code2)

* [Logistic-RegressionvsDecision-Tree](#Logistic-RegressionvsDecision-Tree) | [Typeofdata](#Typeofdata) 

#### Multilayer Perceptron
* [MultilayerPerceptron](#MultilayerPerceptron) | [Code](#Code3)

* [DecisionTreevsMultilayerPerceptron](#DecisionTreevsMultilayerPerceptron)

* [conclusion](#conclusion)

### Decisions-Tree
Decision Trees is a technique that allows analyzing sequential decisions based on the use of results and associated probabilities.
Decision trees can be used to generate expert systems, binary searches and game trees

### Characteristics-of-a-decision-tree
• Poses the problem from different perspectives of action.
• It allows a complete analysis of all possible solutions.
• Provides a scheme to quantify the cost of the result and its probability of use.
• Helps make the best decisions based on existing information and the best assumptions.
• Its structure allows analyzing alternatives, events, probabilities and results.
Within the decision trees we have both advantages and disadvantages of which we have the following.

### Advantage
They are simple to understand and interpret
If the tree is not excessively large, it can be displayed
It does not require overly demanding data preparation (although Scikit-Learn implementation does not support null values)
You can work with both quantitative and qualitative variables
Use a white box model: the algorithm response is easily justifiable from the Boolean logic implemented in it

### Disadvantages
Decision tree learners tend to overtraining, especially when the number of predictive characteristics is high
They are unstable: any small change in the input data can mean a completely different decision tree
It cannot be guaranteed that the tree generated is optimal
There are concepts that are not easily learned because decision trees are not able to express them easily (such as the XOR operator)
Trainees create biased trees if there are dominant classes, so it is recommended to balance the data set before training the trainee

### Code1
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.DecisionTreeClassificationModel
    import org.apache.spark.ml.classification.DecisionTreeClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, 
    VectorIndexer}
    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.sql.SparkSession

    //Se imporortan lod datos del data set Bank
    val data  = spark.read.option("header","true").option("inferSchema", 
    "true").option("delimiter",";").format("csv").load("bank-full.csv")
    // se hace el proceso de categorizar las variables tipo string a numerico
    val yes = 
    data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
    val clean = 
    yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))

    //Se crea un array de los datos seleccionados
    val featureCols = Array("age","previous","balance","duration")
    //Se crea el vector con las columna deatures
    val assembler = new 
    VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    //Se transforma en un nuevo df
    val df2 = assembler.transform(data)
    //Se da un nuevo nobre a la columna y a label
    val featuresLabel = df2.withColumnRenamed("y", "label")
    //Selecciona los indices 
    val dataIndexed = featuresLabel.select("label","features")
    //Se indexan las columnas
    val labelIndexer = new    StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexe
    d)
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
    val evaluator = new
    MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCo
    l("prediction").setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")

    val treeModel = 
    model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
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
    
### Logistic Regression
Logistic regression is a popular method to predict a categorical response. It is a special case of generalized linear models that predicts the probability of the results. In spark.mlla logistic regression can be used to predict a binary result using binomial logistic regression, or it can be used to predict a multiclass result using multinomial logistic regression. Use the family parameter to select between these two algorithms, or leave it unconfigured and Spark will infer the correct variant.

### Logistic regression types
1. Binary logistic regression The categorical response has only two possible outcomes. Example: spam or not
2. Multinomial Logistic Regression
Three or more unordered categories. Example: predict which food is most preferred (Veg, No Veg, Vegan)
3. Ordinary logistic regression gives Three or more categories with orders. Example: movie rating from 1 to 5

### Decision limit
To predict which class a data belongs to, a threshold can be set. Based on this threshold, the estimated probability obtained is classified into classes.
Say, if predicted_value ≥ 0.5, then classify the email as spam, otherwise, not as spam.
The decision limit can be linear or nonlinear. The polynomial order can be increased to obtain complex decision limits.

### Code2
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
    
### Logistic Regression vs Decision Tree
Both algorithms have something in common and is to solve problems through algorithms and can be interpreted easily. Both algorithms have their pros and cons based on proper data management.
Within the comparison we will see if your data is linearly separable
We can see that the Logistic Regression assumes that the data can be separated in a linear or curved way.
Within the decision tree it is not a classifier that can sort the data linearly

### Type of data
Categorical data works well with decision trees, while continuous data works well with logistic regression.
If your data is categorical, then the Logistic Regression cannot handle pure categorical data. On the contrary, you must convert it to numerical data.

### Multilayer Perceptron
The multilayer perceptron classifier (MLPC) is a classifier based on the artificial neural network of direct feeding. MLPC consists of multiple layers of nodes. Each layer is completely connected to the next layer in the network. The nodes in the input layer represent the input data. All other nodes assign inputs to outputs through a linear combination of the inputs with the weights and biases of the node and applying an activation function.
MLPs are useful in research because of their ability to solve problems stochastically, which often allows approximate solutions to extremely complex problems, such as the approach to physical fitness.
MLPs are approximators of universal functions as shown in Cybenko's theorem, so they can be used to create mathematical models through regression analysis. Since classification is a particular case of regression when the response variable is categorical, MLPs are good classification algorithms.
The MLPs were a popular machine learning solution in the 1980s, finding applications in various fields, such as voice recognition, image recognition and machine translation software, but then faced strong competition from vector machines support much simpler (and related). The interest in backpropagation networks returned due to the success of deep learning.
### Code3
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
### Decision Tree vs Multilayer Perceptron
On the difference of the decision tree and multilayer perception have something curious that the decision tree is preferable for the selection of functions and Multilayer Perceptron for classification



### conclusion
There are so many algorithms that can be used to classify our data at my own discretion I lean more towards the Decision Tree algorithm because it is very easy to understand and implement clearly that it does not have a linear order as logical regression but by speed of understanding and application it is An optimal algorithm for my use.

for the infomation about the graphics here below is the link where you can check it
https://docs.google.com/document/d/137MM0gi7ayva83rYYVqD6NmRodsqR6pLgDwj11rg74k/edit#

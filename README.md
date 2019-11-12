## Unidad 2
---
### Practice1
* [Correlation](#Correlation) | [HypothesisTesting](#HypothesingTesting) | [Summarizer](#Summarizer) 
### Practice2
* [DecisionTreeClassifier](#DecisionTreeClassifier)
### Practice3
* [RandomForestClassifier](#RandomForestClassifier)
### Practice4
* [Gradienr-boostedTreeClassifier](#Gradienr-boostedTreeClassifier)
### Practice5
* [MultilayerPerceptronClassifier](#MultilayerPerceptronClassifier)
### Practice6
* [LinearSupportVectorMachine](#LinearSupportVectorMachine)
### Practice7
* [One-vs-RestClassifer](#One-vs-RestClassifer)
### Practice8
* [NaiveBayes](#NaiveBayes)
### Practice1
#### Correlation
Calculating the correlation between two series of data is a common operation in Statistics. In spark.ml we provide the flexibility to calculate pairwise correlations among many series. The supported correlation methods are currently Pearson’s and Spearman’s correlation.

    import org.apache.spark.ml.linalg.{Matrix, Vectors}
    import org.apache.spark.ml.stat.Correlation
    import org.apache.spark.sql.Row

    val data = Seq(Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),Vectors.dense(4.0, 5.0, 0.0, 3.0),Vectors.dense(6.0, 7.0, 0.0, 8.0),Vectors.sparse(4, Seq((0, 9.0), (3, 1.0))))

    val df = data.map(Tuple1.apply).toDF("features")

    val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
    println(s"Pearson correlation matrix:\n $coeff1")

    val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
    println(s"Spearman correlation matrix:\n $coeff2")

#### HypothesisTesting
Hypothesis testing is a powerful tool in statistics to determine whether a result is statistically significant, whether this result occurred by chance or not.

    import org.apache.spark.ml.linalg.{Vector, Vectors}
    import org.apache.spark.ml.stat.ChiSquareTest


    val data = Seq(
      (0.0, Vectors.dense(0.5, 10.0)),
      (0.0, Vectors.dense(1.5, 20.0)),
      (1.0, Vectors.dense(1.5, 30.0)),
      (0.0, Vectors.dense(3.5, 30.0)),
      (0.0, Vectors.dense(3.5, 40.0)),
      (1.0, Vectors.dense(3.5, 40.0))
    )

    val df = data.toDF("label", "features")

    val chi = ChiSquareTest.test(df, "features", "label").head

    println(s"pValues = ${chi.getAs[Vector](0)}")
    println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
    println(s"statistics ${chi.getAs[Vector](2)}")    
    
#### Summarizer
The following example demonstrates using Summarizer to compute the mean and variance for a vector column of the input dataframe, with and without a weight column.

    import org.apache.spark.ml.linalg.{Vector, Vectors}
    import org.apache.spark.ml.stat.Summarizer
    import org.apache.spark.sql.SparkSession


    val spark = SparkSession.builder.appName("SummarizerExample").getOrCreate()

    import spark.implicits._
    import Summarizer._

    val data = Seq((Vectors.dense(2.0, 3.0, 5.0), 1.0), (Vectors.dense(4.0, 6.0, 7.0), 2.0))

    val df = data.toDF("features", "weight")

    val (meanVal, varianceVal) = df.select(metrics("mean", "variance").summary($"features",$"weight").as("summary")).select("summary.mean", "summary.variance").as[(Vector, Vector)].first()

    println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")

    val (meanVal2, varianceVal2) = df.select(mean($"features"), variance($"features")).as[(Vector, Vector)].first()

    println(s"without weight: mean = ${meanVal2}, sum = ${varianceVal2}")

    spark.stop()
### Practice2
#### DecisionTreeClassifier
The following examples load a dataset in LibSVM format, split it into training and test sets, train on the first dataset, and then evaluate on the held-out test set. We use two feature transformers to prepare the data; these help index categories for the label and categorical features, adding metadata to the DataFrame which the Decision Tree algorithm can recognize.
    
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.DecisionTreeClassificationModel
    import org.apache.spark.ml.classification.DecisionTreeClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

    val data = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")

    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
    // 
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data) // features with > 4 distinct values are treated as continuous.

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)


    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    predictions.select("predictedLabel", "label", "features").show(5)

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)

    println(s"Test Error = ${(1.0 - accuracy)}")


    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]

    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
    
    
### Practice3
#### RandomForestClassifier
The following examples load a dataset in LibSVM format, split it into training and test sets, train on the first dataset, and then evaluate on the held-out test set. We use two feature transformers to prepare the data; these help index categories for the label and categorical features, adding metadata to the DataFrame which the tree-based algorithms can recognize.

    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

    val data = spark.read.format("libsvm").load("./sample_libsvm_data.txt") 
    data.show()
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    predictions.select("predictedLabel", "label", "features").show(5)

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println(s"Learned classification forest model:\n ${rfModel.toDebugString}")

### Practice4
#### Gradienr-boostedTreeClassifier
The following examples load a dataset in LibSVM format, split it into training and test sets, train on the first dataset, and then evaluate on the held-out test set. We use two feature transformers to prepare the data; these help index categories for the label and categorical features, adding metadata to the DataFrame which the tree-based algorithms can recognize.

    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

    val data = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")

    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")

    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    predictions.select("predictedLabel", "label", "features").show(5)

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${1.0 - accuracy}")

    val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
    println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")

### Practice5
#### MultilayerPerceptronClassifier
Multilayer perceptron classifier (MLPC) is a classifier based on the feedforward artificial neural network. MLPC consists of multiple layers of nodes. Each layer is fully connected to the next layer in the network. Nodes in the input layer represent the input data. All other nodes map inputs to outputs by a linear combination of the inputs with the node’s weights w w and bias b and applying an activation function.

    import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

    val data = spark.read.format("libsvm").load("data/sample_multiclass_classification_data.txt")


    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)


    val layers = Array[Int](4, 5, 4, 3)

    val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)


    val model = trainer.fit(train)


    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")

    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

### Practice6
#### LinearSupportVectorMachine
A support vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks. Intuitively, a good separation is achieved by the hyperplane that has the largest distance to the nearest training-data points of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier. LinearSVC in Spark ML supports binary classification with linear SVM. Internally, it optimizes the Hinge Loss using OWLQN optimizer.

    import org.apache.spark.ml.classification.LinearSVC

    val training = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")

    val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

    val lsvcModel = lsvc.fit(training)

    println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

### Practice7
#### One-vs-RestClassifer
The example below demonstrates how to load the Iris dataset, parse it as a DataFrame and perform multiclass classification using OneVsRest. The test error is calculated to measure the algorithm accuracy.

    import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

    val inputData = spark.read.format("libsvm").load("data/sample_multiclass_classification_data.txt")

    val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))

    val classifier = new LogisticRegression().setMaxIter(10).setTol(1E-6).setFitIntercept(true)

    val ovr = new OneVsRest().setClassifier(classifier)

    val ovrModel = ovr.fit(train)

    val predictions = ovrModel.transform(test)

    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${1 - accuracy}")

### Practice8
#### NaiveBayes
Naive Bayes is a simple multiclass classification algorithm with the assumption of independence between every pair of features. Naive Bayes can be trained very efficiently. Within a single pass to the training data, it computes the conditional probability distribution of each feature given label, and then it applies Bayes’ theorem to compute the conditional probability distribution of label given an observation and use it for prediction.

    import org.apache.spark.ml.classification.NaiveBayes
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

    val data = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

    val model = new NaiveBayes().fit(trainingData)

    val predictions = model.transform(testData)
    predictions.show()

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy = $accuracy")
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

### Output

    scala> :load Practica1_3.scala
    Loading Practica1_3.scala...
    import org.apache.spark.ml.linalg.{Vector, Vectors}
    import org.apache.spark.ml.stat.Summarizer
    import org.apache.spark.sql.SparkSession
    19/11/12 16:41:15 WARN SparkSession$Builder: Using an existing SparkSession; some configuration may not take effect.
    spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@e958ef
    import spark.implicits._
    import Summarizer._
    data: Seq[(org.apache.spark.ml.linalg.Vector, Double)] = List(([2.0,3.0,5.0],1.0), ([4.0,6.0,7.0],2.0))
    df: org.apache.spark.sql.DataFrame = [features: vector, weight: double]
    meanVal: org.apache.spark.ml.linalg.Vector = [3.333333333333333,5.0,6.333333333333333]
    varianceVal: org.apache.spark.ml.linalg.Vector = [2.0,4.5,2.0]
    with weight: mean = [3.333333333333333,5.0,6.333333333333333], variance = [2.0,4.5,2.0]
    meanVal2: org.apache.spark.ml.linalg.Vector = [3.0,4.5,6.0]
    varianceVal2: org.apache.spark.ml.linalg.Vector = [2.0,4.5,2.0]
    without weight: mean = [3.0,4.5,6.0], sum = [2.0,4.5,2.0]

    scala>

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

### Output

    scala> :load Practica1_2.scala
    Loading Practica1_2.scala...
    import org.apache.spark.ml.linalg.{Vector, Vectors}
    import org.apache.spark.ml.stat.ChiSquareTest
    data: Seq[(Double, org.apache.spark.ml.linalg.Vector)] = List((0.0,[0.5,10.0]), (0.0,[1.5,20.0]), (1.0,[1.5,30.0]), (0.0,[3.5,30.0]), (0.0,[3.5,40.0]), (1.0,[3.5,40.0]))
    df: org.apache.spark.sql.DataFrame = [label: double, features: vector]
    chi: org.apache.spark.sql.Row = [[0.6872892787909721,0.6822703303362126],WrappedArray(2, 3),[0.75,1.5]]
    pValues = [0.6872892787909721,0.6822703303362126]
    degreesOfFreedom [2,3]
    statistics [0.75,1.5]

    scala>

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

### Output

    scala> :load Practica1_3.scala
    Loading Practica1_3.scala...
    import org.apache.spark.ml.linalg.{Vector, Vectors}
    import org.apache.spark.ml.stat.Summarizer
    import org.apache.spark.sql.SparkSession
    19/11/12 16:41:15 WARN SparkSession$Builder: Using an existing SparkSession; some configuration may not take effect.
    spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@e958ef
    import spark.implicits._
    import Summarizer._
    data: Seq[(org.apache.spark.ml.linalg.Vector, Double)] = List(([2.0,3.0,5.0],1.0), ([4.0,6.0,7.0],2.0))
    df: org.apache.spark.sql.DataFrame = [features: vector, weight: double]
    meanVal: org.apache.spark.ml.linalg.Vector = [3.333333333333333,5.0,6.333333333333333]
    varianceVal: org.apache.spark.ml.linalg.Vector = [2.0,4.5,2.0]
    with weight: mean = [3.333333333333333,5.0,6.333333333333333], variance = [2.0,4.5,2.0]
    meanVal2: org.apache.spark.ml.linalg.Vector = [3.0,4.5,6.0]
    varianceVal2: org.apache.spark.ml.linalg.Vector = [2.0,4.5,2.0]
    without weight: mean = [3.0,4.5,6.0], sum = [2.0,4.5,2.0]

    scala>

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
    
  ### Output

    scala> :load Practica_2.scala
    Loading Practica_2.scala...
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.DecisionTreeClassificationModel
    import org.apache.spark.ml.classification.DecisionTreeClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
    19/11/12 16:29:19 WARN LibSVMFileFormat: 'numFeatures' option not specified, determining the number of features by going though the input. If you know the number in advance, please specify it via 'numFeatures' option to avoid the extra scan.
    data: org.apache.spark.sql.DataFrame = [label: double, features: vector]
    labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_86600bffde4c
    featureIndexer: org.apache.spark.ml.feature.VectorIndexerModel = vecIdx_219d3bb62e8e
    trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
    testData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
    dt: org.apache.spark.ml.classification.DecisionTreeClassifier = dtc_3f35579a7ebb
    labelConverter: org.apache.spark.ml.feature.IndexToString = idxToStr_bbd8d688806c
    pipeline: org.apache.spark.ml.Pipeline = pipeline_56f817c5793c
    model: org.apache.spark.ml.PipelineModel = pipeline_56f817c5793c
    predictions: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 6 more fields]
    +--------------+-----+--------------------+
    |predictedLabel|label|            features|
    +--------------+-----+--------------------+
    |           0.0|  0.0|(692,[95,96,97,12...|
    |           0.0|  0.0|(692,[98,99,100,1...|
    |           0.0|  0.0|(692,[122,123,148...|
    |           0.0|  0.0|(692,[123,124,125...|
    |           0.0|  0.0|(692,[123,124,125...|
    +--------------+-----+--------------------+
    only showing top 5 rows

    evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_d4ab771154bb
    accuracy: Double = 0.9411764705882353
    Test Error = 0.05882352941176472
    treeModel: org.apache.spark.ml.classification.DecisionTreeClassificationModel = DecisionTreeClassificationModel (uid=dtc_3f35579a7ebb) of depth 1 with 3 nodes
    Learned classification tree model:
    DecisionTreeClassificationModel (uid=dtc_3f35579a7ebb) of depth 1 with 3 nodes
    If (feature 406 <= 22.0)
    Predict: 1.0
    Else (feature 406 > 22.0)
    Predict: 0.0
    scala>

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

### Output

    scala> :load Practica_3.scala
    Loading Practica_3.scala...
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
    org.apache.spark.sql.AnalysisException: Path does not exist: file:/home/sergio/Documents/Datos_Masivos/Big_Data/Practicas/Unidad2/sample_libsvm_data.txt;
    at org.apache.spark.sql.execution.datasources.DataSource$$anonfun$org$apache$spark$sql$execution$datasources$DataSource$$checkAndGlobPathIfNecessary$1.apply(DataSource.scala:558)
    at org.apache.spark.sql.execution.datasources.DataSource$$anonfun$org$apache$spark$sql$execution$datasources$DataSource$$checkAndGlobPathIfNecessary$1.apply(DataSource.scala:545)
    at scala.collection.TraversableLike$$anonfun$flatMap$1.apply(TraversableLike.scala:241)
    at scala.collection.TraversableLike$$anonfun$flatMap$1.apply(TraversableLike.scala:241)
    at scala.collection.immutable.List.foreach(List.scala:392)
    at scala.collection.TraversableLike$class.flatMap(TraversableLike.scala:241)
    at scala.collection.immutable.List.flatMap(List.scala:355)
    at org.apache.spark.sql.execution.datasources.DataSource.org$apache$spark$sql$execution$datasources$DataSource$$checkAndGlobPathIfNecessary(DataSource.scala:545)
    at org.apache.spark.sql.execution.datasources.DataSource.resolveRelation(DataSource.scala:359)
    at org.apache.spark.sql.DataFrameReader.loadV1Source(DataFrameReader.scala:223)
    at org.apache.spark.sql.DataFrameReader.load(DataFrameReader.scala:211)
    at org.apache.spark.sql.DataFrameReader.load(DataFrameReader.scala:178)
    ... 81 elided
    +-----+--------------------+
    |label|            features|
    +-----+--------------------+
    |  0.0|(692,[127,128,129...|
    |  1.0|(692,[158,159,160...|
    |  1.0|(692,[124,125,126...|
    |  1.0|(692,[152,153,154...|
    |  1.0|(692,[151,152,153...|
    |  0.0|(692,[129,130,131...|
    |  1.0|(692,[158,159,160...|
    |  1.0|(692,[99,100,101,...|
    |  0.0|(692,[154,155,156...|
    |  0.0|(692,[127,128,129...|
    |  1.0|(692,[154,155,156...|
    |  0.0|(692,[153,154,155...|
    |  0.0|(692,[151,152,153...|
    |  1.0|(692,[129,130,131...|
    |  0.0|(692,[154,155,156...|
    |  1.0|(692,[150,151,152...|
    |  0.0|(692,[124,125,126...|
    |  0.0|(692,[152,153,154...|
    |  1.0|(692,[97,98,99,12...|
    |  1.0|(692,[124,125,126...|
    +-----+--------------------+
    only showing top 20 rows

    labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_53a73f032efd
    featureIndexer: org.apache.spark.ml.feature.VectorIndexerModel = vecIdx_3b070c92db4a
    trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
    testData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
    rf: org.apache.spark.ml.classification.RandomForestClassifier = rfc_57c3a7647ff1
    labelConverter: org.apache.spark.ml.feature.IndexToString = idxToStr_db97234b0316
    pipeline: org.apache.spark.ml.Pipeline = pipeline_ed30391dd8f4
    model: org.apache.spark.ml.PipelineModel = pipeline_ed30391dd8f4
    predictions: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 6 more fields]
    +--------------+-----+--------------------+
    |predictedLabel|label|            features|
    +--------------+-----+--------------------+
    |           0.0|  0.0|(692,[98,99,100,1...|
    |           0.0|  0.0|(692,[122,123,148...|
    |           0.0|  0.0|(692,[124,125,126...|
    |           0.0|  0.0|(692,[126,127,128...|
    |           0.0|  0.0|(692,[127,128,129...|
    +--------------+-----+--------------------+
    only showing top 5 rows

    evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_f5dd3759b684
    accuracy: Double = 1.0
    Test Error = 0.0
    rfModel: org.apache.spark.ml.classification.RandomForestClassificationModel = RandomForestClassificationModel (uid=rfc_57c3a7647ff1) with 10 trees
    Learned classification forest model:
    RandomForestClassificationModel (uid=rfc_57c3a7647ff1) with 10 trees
    Tree 0 (weight 1.0):
        If (feature 510 <= 6.5)
        If (feature 458 <= 40.0)
        Predict: 0.0
        Else (feature 458 > 40.0)
        If (feature 602 <= 5.0)
        Predict: 0.0
        Else (feature 602 > 5.0)
        Predict: 1.0
        Else (feature 510 > 6.5)
        Predict: 1.0
    Tree 1 (weight 1.0):
        If (feature 463 <= 2.0)
        If (feature 625 <= 15.5)
        If (feature 489 <= 1.5)
        Predict: 1.0
        Else (feature 489 > 1.5)
        Predict: 0.0
        Else (feature 625 > 15.5)
        Predict: 1.0
        Else (feature 463 > 2.0)
        Predict: 0.0
    Tree 2 (weight 1.0):
        If (feature 540 <= 87.0)
        If (feature 578 <= 9.0)
        Predict: 0.0
        Else (feature 578 > 9.0)
        If (feature 183 <= 36.5)
        Predict: 1.0
        Else (feature 183 > 36.5)
        Predict: 0.0
        Else (feature 540 > 87.0)
        Predict: 1.0
    Tree 3 (weight 1.0):
        If (feature 518 <= 6.0)
        If (feature 154 <= 1.0)
        Predict: 0.0
        Else (feature 154 > 1.0)
        Predict: 1.0
        Else (feature 518 > 6.0)
        If (feature 427 <= 6.0)
        Predict: 0.0
        Else (feature 427 > 6.0)
        Predict: 1.0
    Tree 4 (weight 1.0):
        If (feature 429 <= 7.0)
        If (feature 545 <= 1.0)
        If (feature 323 <= 251.5)
        Predict: 1.0
        Else (feature 323 > 251.5)
        Predict: 0.0
        Else (feature 545 > 1.0)
        Predict: 0.0
        Else (feature 429 > 7.0)
        Predict: 1.0
    Tree 5 (weight 1.0):
        If (feature 462 <= 62.5)
        Predict: 1.0
        Else (feature 462 > 62.5)
        Predict: 0.0
    Tree 6 (weight 1.0):
        If (feature 512 <= 1.5)
        If (feature 523 <= 5.0)
        Predict: 0.0
        Else (feature 523 > 5.0)
        Predict: 1.0
        Else (feature 512 > 1.5)
        Predict: 1.0
    Tree 7 (weight 1.0):
        If (feature 512 <= 1.5)
        If (feature 510 <= 6.5)
        Predict: 0.0
        Else (feature 510 > 6.5)
        Predict: 1.0
        Else (feature 512 > 1.5)
        Predict: 1.0
    Tree 8 (weight 1.0):
        If (feature 462 <= 62.5)
        Predict: 1.0
        Else (feature 462 > 62.5)
        Predict: 0.0
    Tree 9 (weight 1.0):
        If (feature 377 <= 34.5)
        If (feature 380 <= 6.5)
        Predict: 1.0
        Else (feature 380 > 6.5)
        Predict: 0.0
        Else (feature 377 > 34.5)
        If (feature 540 <= 62.0)
        Predict: 0.0
        Else (feature 540 > 62.0)
        Predict: 1.0

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

### Output

    scala> :load Practica_4.scala
    Loading Practica_4.scala...
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
    19/11/12 16:31:38 WARN LibSVMFileFormat: 'numFeatures' option not specified, determining the number of features by going though the input. If you know the number in advance, please specify it via 'numFeatures' option to avoid the extra scan.
    data: org.apache.spark.sql.DataFrame = [label: double, features: vector]
    labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_2f6840f25c29
    featureIndexer: org.apache.spark.ml.feature.VectorIndexerModel = vecIdx_c663ebd92ee3
    trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
    testData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
    gbt: org.apache.spark.ml.classification.GBTClassifier = gbtc_08d4eb448cff
    labelConverter: org.apache.spark.ml.feature.IndexToString = idxToStr_23abbd8a6ac1
    pipeline: org.apache.spark.ml.Pipeline = pipeline_826260bcfff7
    model: org.apache.spark.ml.PipelineModel = pipeline_826260bcfff7
    predictions: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 6 more fields]
    +--------------+-----+--------------------+
    |predictedLabel|label|            features|
    +--------------+-----+--------------------+
    |           0.0|  0.0|(692,[121,122,123...|
    |           0.0|  0.0|(692,[122,123,148...|
    |           0.0|  0.0|(692,[123,124,125...|
    |           0.0|  0.0|(692,[123,124,125...|
    |           0.0|  0.0|(692,[124,125,126...|
    +--------------+-----+--------------------+
    only showing top 5 rows

    evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_618c6f8c0a0e
    accuracy: Double = 0.9696969696969697
    Test Error = 0.030303030303030276
    gbtModel: org.apache.spark.ml.classification.GBTClassificationModel = GBTClassificationModel (uid=gbtc_08d4eb448cff) with 10 trees
    Learned classification GBT model:
    GBTClassificationModel (uid=gbtc_08d4eb448cff) with 10 trees
    Tree 0 (weight 1.0):
        If (feature 406 <= 126.5)
        If (feature 99 in {2.0})
        Predict: -1.0
        Else (feature 99 not in {2.0})
        Predict: 1.0
        Else (feature 406 > 126.5)
        Predict: -1.0
    Tree 1 (weight 0.1):
        If (feature 490 <= 27.5)
        If (feature 239 <= 253.5)
        Predict: 0.47681168808847024
        Else (feature 239 > 253.5)
        Predict: -0.4768116880884694
        Else (feature 490 > 27.5)
        Predict: -0.47681168808847013
    Tree 2 (weight 0.1):
        If (feature 434 <= 90.5)
        If (feature 595 <= 253.5)
        Predict: 0.4381935810427206
        Else (feature 595 > 253.5)
        Predict: -0.43819358104271977
        Else (feature 434 > 90.5)
        If (feature 351 <= 1.0)
        Predict: -0.4381935810427206
        Else (feature 351 > 1.0)
        Predict: -0.4381935810427207
    Tree 3 (weight 0.1):
        If (feature 490 <= 27.5)
        If (feature 239 <= 253.5)
        If (feature 156 <= 23.0)
        If (feature 95 in {0.0})
            Predict: 0.4051496802845983
        Else (feature 95 not in {0.0})
            Predict: 0.4051496802845984
        Else (feature 156 > 23.0)
        If (feature 345 <= 6.5)
            If (feature 180 <= 10.0)
            Predict: 0.4051496802845983
            Else (feature 180 > 10.0)
            Predict: 0.4051496802845984
        Else (feature 345 > 6.5)
            Predict: 0.4051496802845983
        Else (feature 239 > 253.5)
        Predict: -0.4051496802845982
        Else (feature 490 > 27.5)
        If (feature 404 <= 1.0)
        If (feature 266 <= 28.0)
        If (feature 128 <= 1.0)
            Predict: -0.4051496802845983
        Else (feature 128 > 1.0)
            Predict: -0.4051496802845984
        Else (feature 266 > 28.0)
        Predict: -0.40514968028459836
        Else (feature 404 > 1.0)
        Predict: -0.40514968028459836
    Tree 4 (weight 0.1):
        If (feature 433 <= 52.5)
        If (feature 295 <= 253.5)
        If (feature 129 <= 181.0)
        Predict: 0.37658413183529915
        Else (feature 129 > 181.0)
        Predict: 0.3765841318352994
        Else (feature 295 > 253.5)
        Predict: -0.3765841318352994
        Else (feature 433 > 52.5)
        If (feature 124 <= 14.5)
        Predict: -0.37658413183529926
        Else (feature 124 > 14.5)
        Predict: -0.3765841318352994
    Tree 5 (weight 0.1):
        If (feature 406 <= 126.5)
        If (feature 568 <= 253.5)
        Predict: 0.35166478958101
        Else (feature 568 > 253.5)
        Predict: -0.3516647895810099
        Else (feature 406 > 126.5)
        If (feature 599 <= 198.5)
        Predict: -0.35166478958101005
        Else (feature 599 > 198.5)
        Predict: -0.3516647895810101
    Tree 6 (weight 0.1):
        If (feature 406 <= 126.5)
        If (feature 295 <= 253.5)
        Predict: 0.32974984655529926
        Else (feature 295 > 253.5)
        Predict: -0.32974984655530015
        Else (feature 406 > 126.5)
        If (feature 407 <= 160.0)
        Predict: -0.32974984655529926
        Else (feature 407 > 160.0)
        If (feature 629 <= 184.5)
        Predict: -0.32974984655529926
        Else (feature 629 > 184.5)
        Predict: -0.3297498465552995
    Tree 7 (weight 0.1):
        If (feature 434 <= 90.5)
        If (feature 549 <= 253.5)
        If (feature 625 <= 118.5)
        Predict: 0.31033724551979563
        Else (feature 625 > 118.5)
        Predict: 0.3103372455197957
        Else (feature 549 > 253.5)
        Predict: -0.31033724551979525
        Else (feature 434 > 90.5)
        Predict: -0.3103372455197956
    Tree 8 (weight 0.1):
        If (feature 406 <= 126.5)
        If (feature 627 <= 2.5)
        Predict: -0.2930291649125432
        Else (feature 627 > 2.5)
        Predict: 0.2930291649125433
        Else (feature 406 > 126.5)
        If (feature 461 <= 46.5)
        Predict: -0.2930291649125433
        Else (feature 461 > 46.5)
        If (feature 461 <= 127.5)
        Predict: -0.2930291649125433
        Else (feature 461 > 127.5)
        Predict: -0.29302916491254344
    Tree 9 (weight 0.1):
        If (feature 433 <= 52.5)
        If (feature 268 <= 253.5)
        If (feature 185 <= 193.5)
        If (feature 126 <= 16.0)
            Predict: 0.27750666438358246
        Else (feature 126 > 16.0)
            Predict: 0.2775066643835825
        Else (feature 185 > 193.5)
        Predict: 0.27750666438358257
        Else (feature 268 > 253.5)
        Predict: -0.2775066643835826
        Else (feature 433 > 52.5)
        If (feature 267 <= 82.0)
        Predict: -0.2775066643835825
        Else (feature 267 > 82.0)
        If (feature 156 <= 41.5)
        Predict: -0.27750666438358246
        Else (feature 156 > 41.5)
        Predict: -0.2775066643835826


    scala>


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

### Output

    scala> :load Practica_5.scala
    Loading Practica_5.scala...
    import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    19/11/12 16:33:12 WARN LibSVMFileFormat: 'numFeatures' option not specified, determining the number of features by going though the input. If you know the number in advance, please specify it via 'numFeatures' option to avoid the extra scan.
    data: org.apache.spark.sql.DataFrame = [label: double, features: vector]
    splits: Array[org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]] = Array([label: double, features: vector], [label: double, features: vector])
    train: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
    test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
    layers: Array[Int] = Array(4, 5, 4, 3)
    trainer: org.apache.spark.ml.classification.MultilayerPerceptronClassifier = mlpc_9295473eaace
    model: org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel = mlpc_9295473eaace
    result: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 3 more fields]
    predictionAndLabels: org.apache.spark.sql.DataFrame = [prediction: double, label: double]
    evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_8b6af532db9e
    Test set accuracy = 0.9019607843137255

### Practice6
#### LinearSupportVectorMachine
A support vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks. Intuitively, a good separation is achieved by the hyperplane that has the largest distance to the nearest training-data points of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier. LinearSVC in Spark ML supports binary classification with linear SVM. Internally, it optimizes the Hinge Loss using OWLQN optimizer.

    import org.apache.spark.ml.classification.LinearSVC

    val training = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")

    val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

    val lsvcModel = lsvc.fit(training)

    println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

### Output

    scala> :load Practica_6.scala
    Loading Practica_6.scala...
    import org.apache.spark.ml.classification.LinearSVC
    19/11/12 16:34:00 WARN LibSVMFileFormat: 'numFeatures' option not specified, determining the number of features by going though the input. If you know the number in advance, please specify it via 'numFeatures' option to avoid the extra scan.
    training: org.apache.spark.sql.DataFrame = [label: double, features: vector]
    lsvc: org.apache.spark.ml.classification.LinearSVC = linearsvc_18e1df2bd67a
    lsvcModel: org.apache.spark.ml.classification.LinearSVCModel = linearsvc_18e1df2bd67a
    Coefficients: [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-5.170630317473439E-4,-1.172288654973735E-4,-8.882754836918948E-5,8.522360710187464E-5,0.0,0.0,-1.3436361263314267E-5,3.729569801338091E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,8.888949552633658E-4,2.9864059761812683E-4,3.793378816193159E-4,-1.762328898254081E-4,0.0,1.5028489269747836E-6,1.8056041144946687E-6,1.8028763260398597E-6,-3.3843713506473646E-6,-4.041580184807502E-6,2.0965017727015125E-6,8.536111642989494E-5,2.2064177429604464E-4,2.1677599940575452E-4,-5.472401396558763E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,9.21415502407147E-4,3.1351066886882195E-4,2.481984318412822E-4,0.0,-4.147738197636148E-5,-3.6832150384497175E-5,0.0,-3.9652366184583814E-6,-5.1569169804965594E-5,-6.624697287084958E-5,-2.182148650424713E-5,1.163442969067449E-5,-1.1535211416971104E-6,3.8138960488857075E-5,1.5823711634321492E-6,-4.784013432336632E-5,-9.386493224111833E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.3174897827077767E-4,1.7055492867397665E-4,0.0,-2.7978204136148868E-5,-5.88745220385208E-5,-4.1858794529775E-5,-3.740692964881002E-5,-3.9787939304887E-5,-5.545881895011037E-5,-4.505015598421474E-5,-3.214002494749943E-6,-1.6561868808274739E-6,-4.416063987619447E-6,-7.9986183315327E-6,-4.729962112535003E-5,-2.516595625914463E-5,-3.6407809279248066E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.4719098130614967E-4,0.0,-3.270637431382939E-5,-5.5703407875748054E-5,-5.2336892125702286E-5,-7.829604482365818E-5,-7.60385448387619E-5,-8.371051301348216E-5,-1.8669558753795108E-5,0.0,1.2045309486213725E-5,-2.3374084977016397E-5,-1.0788641688879534E-5,-5.5731194431606874E-5,-7.952979033591137E-5,-1.4529196775456057E-5,8.737948348132623E-6,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0012589360772978808,-1.816228630214369E-4,-1.0650711664557365E-4,-6.040355527710781E-5,-4.856392973921569E-5,-8.973895954652451E-5,-8.78131677062384E-5,-5.68487774673792E-5,-3.780926734276347E-5,1.3834897036553787E-5,7.585485129441565E-5,5.5017411816753975E-5,-1.5430755398169695E-5,-1.834928703625931E-5,-1.0354008265646844E-4,-1.3527847721351194E-4,-1.1245007647684532E-4,-2.9373916056750564E-5,-7.311217847336934E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.858228613863785E-4,-1.2998173971449976E-4,-1.478408021316135E-4,-8.203374605865772E-5,-6.556685320008032E-5,-5.6392660386580244E-5,-6.995571627330911E-5,-4.664348159856693E-5,-2.3026593698824318E-5,7.398833979172035E-5,1.4817176130099997E-4,1.0938317435545486E-4,7.940425167011364E-5,-6.743294804348106E-7,-1.2623302721464762E-4,-1.9110387355357616E-4,-1.8611622108961136E-4,-1.2776766254736952E-4,-8.935302806524433E-5,-1.239417230441996E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.829530831354112E-4,-1.3912189600461263E-4,-1.2593136464577562E-4,-5.964745187930992E-5,-5.360328152341982E-5,-1.0517880662090183E-4,-1.3856124131005022E-4,-7.181032974125911E-5,2.3249038865093483E-6,1.566964269571967E-4,2.3261206954040812E-4,1.7261638232256968E-4,1.3857530960270466E-4,-1.396299028868332E-5,-1.5765773982418597E-4,-2.0728798812007546E-4,-1.9106441272002828E-4,-1.2744834161431415E-4,-1.2755611630280015E-4,-5.1885591560478935E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.59081567023441E-4,-1.216531230287931E-4,-5.623851079809818E-5,-3.877987126382982E-5,-7.550900509956966E-5,-1.0703140005463545E-4,-1.4720428138106226E-4,-8.781423374509368E-5,7.941655609421792E-5,2.3206354986219992E-4,2.7506982343672394E-4,2.546722233188043E-4,1.810821666388498E-4,-1.3069916689929984E-5,-1.842374220886751E-4,-1.977540482445517E-4,-1.7722074063670741E-4,-1.487987014723575E-4,-1.1879021431288621E-4,-9.755283887790393E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.302740311359312E-4,-5.3683030235535024E-5,-1.7631200013656873E-5,-7.846611034608254E-5,-1.22100767283256E-4,-1.7281968533449702E-4,-1.5592346128894157E-4,-5.239579492910452E-5,1.680719343542442E-4,2.8930086786548053E-4,3.629921493231646E-4,2.958223512266975E-4,2.1770466955449064E-4,-6.40884808188951E-5,-1.9058225556007997E-4,-2.0425138564600712E-4,-1.711994903702119E-4,-1.3853486798341369E-4,-1.3018592950855062E-4,-1.1887779512760102E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-7.021411112285498E-5,-1.694500843168125E-5,-7.189722824172193E-5,-1.4560828004346436E-4,-1.4935497340563198E-4,-1.9496419340776972E-4,-1.7383743417254187E-4,-3.3438825792010694E-5,2.866538327947017E-4,2.9812321570739803E-4,3.77250607691119E-4,3.211702827486386E-4,2.577995115175486E-4,-1.6627385656703205E-4,-1.8037105851523224E-4,-2.0419356344211325E-4,-1.7962237203420184E-4,-1.3726488083579862E-4,-1.3461014473741762E-4,-1.2264216469164138E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0015239752514658556,-5.472330865993813E-5,-9.65684394936216E-5,-1.3424729853486994E-4,-1.4727467799568E-4,-1.616270978824712E-4,-1.8458259010029364E-4,-1.9699647135089726E-4,1.3085261294290817E-4,2.943178857107149E-4,3.097773692834126E-4,4.112834769312103E-4,3.4113620757035025E-4,1.6529945924367265E-4,-2.1065410862650534E-4,-1.883924081539624E-4,-1.979586414569358E-4,-1.762131187223702E-4,-1.272343622678854E-4,-1.2708161719220297E-4,-1.4812221011889967E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.001140680600536578,-1.323467421269896E-4,-1.2904607854274846E-4,-1.4104748544921958E-4,-1.5194605434027872E-4,-2.1104539389774283E-4,-1.7911827582001795E-4,-1.8952948277194435E-4,2.1767571552539842E-4,3.0201791656326465E-4,4.002863274397723E-4,4.0322806756364006E-4,4.118077382608461E-4,3.7917405252859545E-6,-1.9886290660234838E-4,-1.9547443112937263E-4,-1.9857348218680872E-4,-1.3336892200703206E-4,-1.2830129292910815E-4,-1.1855916317355505E-4,-1.765597203760205E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0010938769592297973,-1.2785475305234688E-4,-1.3424699777466666E-4,-1.505200652479287E-4,-1.9333287822872713E-4,-2.0385160086594937E-4,-1.7422470698847553E-4,4.63598443910652E-5,2.0617623087127652E-4,2.862882891134514E-4,4.074830988361515E-4,3.726357785147985E-4,3.507520190729629E-4,-1.516485494364312E-4,-1.7053751921469217E-4,-1.9638964654350848E-4,-1.9962586265806435E-4,-1.3612312664311173E-4,-1.218285533892454E-4,-1.1166712081624676E-4,-1.377283888177579E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-3.044386260118809E-4,-1.240836643202059E-4,-1.335317492716633E-4,-1.5783442604618277E-4,-1.9168434243384107E-4,-1.8710322733892716E-4,-1.1283989231463139E-4,1.1136504453105364E-4,1.8707244892705632E-4,2.8654279528966305E-4,4.0032117544983536E-4,3.169637536305377E-4,2.0158994278679014E-4,-1.3139392844616033E-4,-1.5181070482383948E-4,-1.825431845981843E-4,-1.602539928567571E-4,-1.3230404795396355E-4,-1.1669138691257469E-4,-1.0532154964150405E-4,-1.3709037042366007E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-4.0287410145021705E-4,-1.3563987950912995E-4,-1.3225887084018914E-4,-1.6523502389794188E-4,-2.0175074284706945E-4,-1.572459106394481E-4,2.577536501278673E-6,1.312463663419457E-4,2.0707422291927531E-4,3.9081065544314936E-4,3.3487058329898135E-4,2.5790441367156086E-4,2.6881819648016494E-5,-1.511383586714907E-4,-1.605428139328567E-4,-1.7267287462873575E-4,-1.1938943768052963E-4,-1.0505245038633314E-4,-1.1109385509034013E-4,-1.3469914274864725E-4,-2.0735223736035555E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-5.034374233912422E-4,-1.5961213688405883E-4,-1.274222123810994E-4,-1.582821104884909E-4,-2.1301220616286252E-4,-1.2933366375029613E-4,1.6802673102179614E-5,1.1020918082727098E-4,2.1160795272688753E-4,3.4873421050827716E-4,2.6487211944380384E-4,1.151606835026639E-4,-5.4682731396851946E-5,-1.3632001630934325E-4,-1.4340405857651405E-4,-1.248695773821634E-4,-8.462873247977974E-5,-9.580708414770257E-5,-1.0749166605399431E-4,-1.4618038459197777E-4,-3.7556446296204636E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-5.124342611878493E-4,-2.0369734099093433E-4,-1.3626985098328694E-4,-1.3313768183302705E-4,-1.871555537819396E-4,-1.188817315789655E-4,-1.8774817595622694E-5,5.7108412194993384E-5,1.2728161056121406E-4,1.9021458214915667E-4,1.2177397895874969E-4,-1.2461153574281128E-5,-7.553961810487739E-5,-1.0242174559410404E-4,-4.44873554195981E-5,-9.058561577961895E-5,-6.837347198855518E-5,-8.084409304255458E-5,-1.3316868299585082E-4,-2.0335916397646626E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-3.966510928472775E-4,-1.3738983629066386E-4,-3.7971221409699866E-5,-6.431763035574533E-5,-1.1857739882295322E-4,-9.359520863114822E-5,-5.0878371516215046E-5,-8.269367595092908E-8,0.0,1.3434539131099211E-5,-1.9601690213728576E-6,-2.8527045990494954E-5,-7.410332699310603E-5,-7.132130570080122E-5,-4.9780961185536E-5,-6.641505361384578E-5,-6.962005514093816E-5,-7.752898158331023E-5,-1.7393609499225025E-4,-0.0012529479255443958,0.0,0.0,2.0682521269893754E-4,0.0,0.0,0.0,0.0,0.0,-4.6702467383631055E-4,-1.0318036388792008E-4,1.2004408785841247E-5,0.0,-2.5158639357650687E-5,-1.2095240910793449E-5,-5.19052816902203E-6,-4.916790639558058E-6,-8.48395853563783E-6,-9.362757097074547E-6,-2.0959335712838412E-5,-4.7790091043859085E-5,-7.92797600958695E-5,-4.462687041778011E-5,-4.182992428577707E-5,-3.7547996285851254E-5,-4.52754480225615E-5,-1.8553562561513456E-5,-2.4763037962085644E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-3.4886180455242474E-4,-5.687523659359091E-6,7.380040279654313E-5,4.395860636703821E-5,7.145198242379862E-5,6.181248343370637E-6,0.0,-6.0855538083486296E-5,-4.8563908323274725E-5,-4.117920588930435E-5,-4.359283623112936E-5,-6.608754161500044E-5,-5.443032251266018E-5,-2.7782637880987207E-5,0.0,0.0,2.879461393464088E-4,-0.0028955529777851255,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.2312114837837392E-4,-1.9526747917254753E-5,-1.6999506829961688E-5,5.4835294148085086E-5,1.523441632762399E-5,-5.8365604525328614E-5,-1.2378194216521848E-4,-1.1750704953254656E-4,-6.19711523061306E-5,-5.042009645812091E-5,-1.4055260223565886E-4,-1.410330942465528E-4,-1.9272308238929396E-4,-4.802489964676616E-4] Intercept: 0.012911305214513969

    scala>

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


### Output

    scala> :load Practica_7.scala
    Loading Practica_7.scala...
    import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    19/11/12 16:35:35 WARN LibSVMFileFormat: 'numFeatures' option not specified, determining the number of features by going though the input. If you know the number in advance, please specify it via 'numFeatures' option to avoid the extra scan.
    inputData: org.apache.spark.sql.DataFrame = [label: double, features: vector]
    train: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
    test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
    classifier: org.apache.spark.ml.classification.LogisticRegression = logreg_71b8fd5b40be
    ovr: org.apache.spark.ml.classification.OneVsRest = oneVsRest_ae8ba4e5f1bd
    ovrModel: org.apache.spark.ml.classification.OneVsRestModel = oneVsRest_ae8ba4e5f1bd
    predictions: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 2 more fields]
    evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_d4f0f9efa0c8
    accuracy: Double = 0.9285714285714286
    Test Error = 0.0714285714285714

    scala>

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

### Output

    Scala> :load Practica_8.scala
    Loading Practica_8.scala...
    import org.apache.spark.ml.classification.NaiveBayes
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    19/11/12 16:05:19 WARN LibSVMFileFormat: 'numFeatures' option not specified, determining the number of features by going though the input. If you know the number in advance, please specify it via 'numFeatures' option to avoid the extrascan.
    data: org.apache.spark.sql.DataFrame = [label: double, features: vector]
    trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
    testData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
    model: org.apache.spark.ml.classification.NaiveBayesModel = NaiveBayesModel (uid=nb_e394a0d6518b) with 2 classes
    predictions: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 3 more fields]
    19/11/12 16:05:27 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
    19/11/12 16:05:27 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
    +-----+--------------------+--------------------+-----------+----------+
    |label|            features|       rawPrediction|probability|prediction|
    +-----+--------------------+--------------------+-----------+----------+
    |  0.0|(692,[95,96,97,12...|[-173678.60946628...|  [1.0,0.0]|       0.0|
    |  0.0|(692,[98,99,100,1...|[-178107.24302988...|  [1.0,0.0]|       0.0|
    |  0.0|(692,[100,101,102...|[-100020.80519087...|  [1.0,0.0]|       0.0|
    |  0.0|(692,[124,125,126...|[-183521.85526462...|  [1.0,0.0]|       0.0|
    |  0.0|(692,[127,128,129...|[-183004.12461660...|  [1.0,0.0]|       0.0|
    |  0.0|(692,[128,129,130...|[-246722.96394714...|  [1.0,0.0]|       0.0|
    |  0.0|(692,[152,153,154...|[-208696.01108598...|  [1.0,0.0]|       0.0|
    |  0.0|(692,[153,154,155...|[-261509.59951302...|  [1.0,0.0]|       0.0|
    |  0.0|(692,[154,155,156...|[-217654.71748256...|  [1.0,0.0]|       0.0|
    |  0.0|(692,[181,182,183...|[-155287.07585335...|  [1.0,0.0]|       0.0|
    |  1.0|(692,[99,100,101,...|[-145981.83877498...|  [0.0,1.0]|       1.0|
    |  1.0|(692,[100,101,102...|[-147685.13694275...|  [0.0,1.0]|       1.0|
    |  1.0|(692,[123,124,125...|[-139521.98499849...|  [0.0,1.0]|       1.0|
    |  1.0|(692,[124,125,126...|[-129375.46702012...|  [0.0,1.0]|       1.0|
    |  1.0|(692,[126,127,128...|[-145809.08230799...|  [0.0,1.0]|       1.0|
    |  1.0|(692,[127,128,129...|[-132670.15737290...|  [0.0,1.0]|       1.0|
    |  1.0|(692,[128,129,130...|[-100206.72054749...|  [0.0,1.0]|       1.0|
    |  1.0|(692,[129,130,131...|[-129639.09694930...|  [0.0,1.0]|       1.0|
    |  1.0|(692,[129,130,131...|[-143628.65574273...|  [0.0,1.0]|       1.0|
    |  1.0|(692,[129,130,131...|[-129238.74023248...|  [0.0,1.0]|       1.0|
    +-----+--------------------+--------------------+-----------+----------+
    only showing top 20 rows

    evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_da9d2ac100fc
    accuracy: Double = 1.0
    Test set accuracy = 1.0


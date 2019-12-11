# Unit 3
---
* [Kmeans](#Kmeans) | [Output](#K-means-Output)
* [LogisticRegression](#LogisticRegression) | [OutPut](#LogisticRegression-Output)
* [Test](#Test) | [Output](#Test-Output)

### Kmeans
---
            // Start a Spark Session
        import org.apache.spark.sql.SparkSession

        // Optional: Use the following code below to set the Error reporting
        import org.apache.log4j._
        Logger.getLogger("org").setLevel(Level.ERROR)

        // Spark Session
        val spark = SparkSession.builder().getOrCreate()

        // Import clustering Algorithm
        import org.apache.spark.ml.clustering.KMeans

        // Loads data.
        val dataset = spark.read.format("libsvm").load("sample_kmeans_data.txt")
        // val dataset = spark.read.option("header","true").option("inferSchema","true").csv("sample_kmeans_data.txt")

        // Trains a k-means model.
        val kmeans = new KMeans().setK(2).setSeed(1L)
        val model = kmeans.fit(dataset)

        // Evaluate clustering by computing Within Set Sum of Squared Errors.
        val WSSSE = model.computeCost(dataset)
        println(s"Within Set Sum of Squared Errors = $WSSSE")

        // Shows the result.
        println("Cluster Centers: ")
        model.clusterCenters.foreach(println)
### K-means-Output
---
        Loading kmeansExample.scala...
        import org.apache.spark.sql.SparkSession
        import org.apache.log4j._
        spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@4e77f29e
        import org.apache.spark.ml.clustering.KMeans
        dataset: org.apache.spark.sql.DataFrame = [label: double, features: vector]
        kmeans: org.apache.spark.ml.clustering.KMeans = kmeans_65bf8d9d608e
        19/12/10 16:00:24 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
        19/12/10 16:00:24 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
        model: org.apache.spark.ml.clustering.KMeansModel = kmeans_65bf8d9d608e
        warning: there was one deprecation warning; re-run with -deprecation for details
        WSSSE: Double = 0.11999999999994547
        Within Set Sum of Squared Errors = 0.11999999999994547
        Cluster Centers:
        [0.1,0.1,0.1]
        [9.1,9.1,9.1]
### LogisticRegression
---
      // Importacion de las librerias y apis de Logistic Regression
      import org.apache.spark.ml.classification.LogisticRegression
      //Importacion de la libreria de SparkSeccion
      import org.apache.spark.sql.SparkSession
      // Si requerimos la utilizacion del codigo de  Error reporting lo implementamos aunque regularmente es opcional esta accion.

      import org.apache.log4j._
      Logger.getLogger("org").setLevel(Level.ERROR)
      // Creamos la secion de spark. 
      val spark = SparkSession.builder().getOrCreate()
      // Utilice Spark para leer el archivo que se encuentra en un formato csv llamado Advertising.csv

      val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")
      //Visualizacion de los datos de nustro dataset y analizamos como nos son precentados.
      data.show()
      // Imprecion del Schema del DataFrame
      data.printSchema()


      data.head(1)
      // creamos la variable colnames la cual contendra en un arreglo de string la informacion de la primera columna.
      val colnames = data.columns
      //creacion de la variable fristrow la cual contendra el contendido de la primera columna de datos.
      val firstrow = data.head(1)(0)
      println("\n")
      println("Example data row")
      for(ind <- Range(1, colnames.length)){
          println(colnames(ind))
          println(firstrow(ind))
          println("\n")
      }



      //    - Creamos una nueva clolumna llamada "Hour" del Timestamp conteniendo la  "Hour of the click"
      val timedata = data.withColumn("Hour",hour(data("Timestamp")))
      //    - Renombracion de la columna "Clicked on Ad" a "label"
      val logregdata = timedata.select(data("Clicked on Ad").as("label") ,$"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male")
      // Visualizacion del renombramiento de la columna Clicked on Ad a label.
      logregdata.show()
      // Importacion de las librerias y apis de VectorAssembler y Vectors
      import org.apache.spark.ml.feature.VectorAssembler
      import org.apache.spark.ml.linalg.Vectors
      // Creacion de un nuevo objecto VectorAssembler llamado assembler para los feature las cuales seran las columnas restantes del dataset como un arreglo "Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male"
      val assembler = (new VectorAssembler()
                        .setInputCols(Array("Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male"))
                        .setOutputCol("features"))


      // Utilizamos randomSplit para crear datos de entrenamineto con 70% y de prueba 30% con los que estara interactuando nuestro algoritmo 
      val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)


      // Importacion de la libreria de Pipeline
      import org.apache.spark.ml.Pipeline
      // Creacion de un nuevo objeto de  LogisticRegression llamado lr
      val lr = new LogisticRegression()
      // Creacion de un nuevo  pipeline con los elementos: assembler la cual es nuestras factures, lr duentro objeto de LOgisticRegression
      val pipeline = new Pipeline().setStages(Array(assembler, lr))
      // creacion de la variable model la cual contendra el elemento de pipeline el cual contendra un ajuste (fit)  para el conjunto que nos encontramos entrenando

      val model = pipeline.fit(training)

      // Resultado del modelo con la trasformacion de los datos de prueba.

      val results = model.transform(test)
      results.show()

      // Para la utilizacion de Metrics y Evaluation importamos la libreria de  MulticlassMetrics
      import org.apache.spark.mllib.evaluation.MulticlassMetrics

      // Convertimos los resutalos de la prueba (test) en RDD utilizando .as y .rdd

      val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
      // Inicializamos el objeto MulticlassMetrics con el parametro predictionAndLabels.

      val metrics = new MulticlassMetrics(predictionAndLabels)
      // Impresion de la matrix de confucion
      println("Confusion matrix:")
      println(metrics.confusionMatrix)
      // Imprimimos la exactitud de nuestra predicion la cual es de 0.97 la cual es muy buena.
      metrics.accuracy
### LogisticRegression-Output
---
        logregdata: org.apache.spark.sql.DataFrame = [label: int, Daily Time Spent on Site: double ... 5 more fields]
        +-----+------------------------+---+-----------+--------------------+----+----+
        |label|Daily Time Spent on Site|Age|Area Income|Daily Internet Usage|Hour|Male|
        +-----+------------------------+---+-----------+--------------------+----+----+
        |    0|                   68.95| 35|    61833.9|              256.09|   0|   0|
        |    0|                   80.23| 31|   68441.85|              193.77|   1|   1|
        |    0|                   69.47| 26|   59785.94|               236.5|  20|   0|
        |    0|                   74.15| 29|   54806.18|              245.89|   2|   1|
        |    0|                   68.37| 35|   73889.99|              225.58|   3|   0|
        |    0|                   59.99| 23|   59761.56|              226.74|  14|   1|
        |    0|                   88.91| 33|   53852.85|              208.36|  20|   0|
        |    1|                    66.0| 48|   24593.33|              131.76|   1|   1|
        |    0|                   74.53| 30|    68862.0|              221.51|   9|   1|
        |    0|                   69.88| 20|   55642.32|              183.82|   1|   1|
        |    1|                   47.64| 49|   45632.51|              122.02|  20|   0|
        |    0|                   83.07| 37|   62491.01|              230.87|   8|   1|
        |    1|                   69.57| 48|   51636.92|              113.12|   1|   1|
        |    0|                   79.52| 24|   51739.63|              214.23|  21|   0|
        |    1|                   42.95| 33|    30976.0|              143.56|   9|   0|
        |    1|                   63.45| 23|   52182.23|              140.64|   3|   1|
        |    1|                   55.39| 37|   23936.86|              129.41|  19|   0|
        |    0|                   82.03| 41|   71511.08|              187.53|   7|   0|
        |    1|                    54.7| 36|   31087.54|              118.39|   7|   1|
        |    1|                   74.58| 40|   23821.72|              135.51|   4|   1|
        +-----+------------------------+---+-----------+--------------------+----+----+
        only showing top 20 rows
        import org.apache.spark.mllib.evaluation.MulticlassMetrics
        predictionAndLabels: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[135] at rdd at 
        LogisticRegresion.scala:39
        metrics: org.apache.spark.mllib.evaluation.MulticlassMetrics = 
        org.apache.spark.mllib.evaluation.MulticlassMetrics@9d56384
        Confusion matrix:
        146.0  7.0
        1.0    161.0
        res15: Double = 0.9746031746031746

### Test
---
      import org.apache.spark.sql.SparkSession
      import org.apache.spark.ml.feature.VectorAssembler
      import org.apache.spark.ml.linalg.Vectors
      import org.apache.log4j._
      Logger.getLogger("org").setLevel(Level.ERROR)

      import org.apache.spark.ml.clustering.KMeans

      val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("wholesale_customers_data.csv")

      val assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")

      val output = assembler.transform(data)
      val kmeans = new KMeans().setK(3).setSeed(1L)
      val model = kmeans.fit(output)

      // Evaluate clustering by calculate Within Set Sum of Squared Errors.
      val WSSE = model.computeCost(output)
      println(s"Within set sum of Squared Errors = $WSSE")

      // Show results
      println("Cluster Centers: ")
      model.clusterCenters.foreach(println)
### Test-Output
---
        Within set sum of Squared Errors = 8.095172370767671E10
        Cluster Centers:
        [7993.574780058651,4196.803519061584,5837.4926686217,2546.624633431085,2016.2873900293255,1151.4193548387098]
        [9928.18918918919,21513.081081081084,30993.486486486487,2960.4324324324325,13996.594594594595,3772.3243243243246]
        [35273.854838709674,5213.919354838709,5826.096774193548,6027.6612903225805,1006.9193548387096,2237.6290322580644]

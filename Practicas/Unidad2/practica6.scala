import org.apache.spark.ml.classification.LinearSVC

// Cargar datos de entrenamiento
val training = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")
// Se evalua el algoritmo LinearSVC
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

// Ajustar el modelo
val lsvcModel = lsvc.fit(training)

// Imprime los coeficientes e intercepta para svc lineal
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
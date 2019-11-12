//1.- Comienza una simple sesion spark
import org.apache.spark.sql.SparkSession
//2.- Cargue el archivo de Nextflix Stock CSV, haga que spark infiera los tipos e datos.
val spar = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
//3.- Cuales son los nombres de las Columnas?
df.show()
//Date | Open | High | Low | Close | Volume | AdjClose |

//4.-Como es el esquema
df.printSchema()
//5.- Imprime las primeras 5 columnas
df.show(5)
//6.- Usa describe() para aprender sobre el Dataframe
df.describe().show()
//7.-Crea un nuevo Dataframe con una columna nueva llamada "HV Ratio" que es la 
//relacion entre el precio de la columna "High" frente a la columna "Volumen" 
//de acciones negociadas por un dia

val df2 = df.withColumn("HV Ratio", df("High")*df("Volume"))
//------------------------------
//8.-Que dia tuvo el pico mas alto en la columna "Price"?
//Segun Vicente dice que en el esquema no existe.
//------------------------------

//9.- Cual es el significado de la columna cerrar "Close"?
//A lo qe veo es el cierre de los datos de la columna suena redundante pero sus valores son similares a los de la columna "Open"
//10.-Cual es el maximo y minimo de la columna "Volumen"?
df.select(max("Volume")).show()
df.select(min("Volume")).show()
//Min: 3531300
//Max: 315541800
//11.- Con sintaxis Scala/Spark $ conteste los siguiente:
//a
df.filter($"Close" < 600).count()
//b
val step = df.select($"High").filter($"High" > 500).count()
val result = (step*1.0) / 100.00
//c
var total = df.select(corr($"high",$"Volume")).show()
//d
df.groupBy(year(df("Date"))).max().show()
//e
df.groupBy(month(df("Date"))).avg().show()

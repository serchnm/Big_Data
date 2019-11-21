import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("CitiGroup2006_2008")

df.printSchema()

//1. Columns(i: Iavgnt)
df.filter($"Close" < 480 && $"High" < 480).columns()

//2. Count()
df.filter($"Close" < 480 && $"High" < 480).count()

//3. Sort()
df.sort($"Date".desc).show()

//4.Take()
df.select($"Date").take(5)

//5. CollectAsList()
df.filter($"Close" < 480 && $"High" < 480).collectAsList()

//6. First()
df.filter($"Close" < 480 && $"High" < 480).first()

//7. groupBy()
df.filter($"Close" < 480 && $"High" < 480).groupBy("Close")

//8 dayofweek()
df.select(dayofweek(df("Date"))).show()

//9 dayofmonth()
df.select(dayofmonth(df("Date"))).show()

//10 dayofyear()
df.select(dayofyear(df("Date"))).show()

//11 weekofyear()
df.select(weekofyear(df("Date"))).show()

//12 last_day()
df.select(last_day(df("Date"))).show()

//13 repartition()
df.select("Low").repartition().show()

//14 distinct()
df.select("Date").distinct().show()

//15 sort()
df.sort($"High".asc).show()

//16 min()
df.select(min("High")).show()

//17 mean()
df.select(mean("High")).show()

//18 take()
df.select($"Open").take(1)

//19 max
df.select(max("High")).show()

//20 avg
df.select(avg("High")).show()
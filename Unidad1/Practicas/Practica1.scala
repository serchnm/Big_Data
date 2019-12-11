
//1. Desarrollar un algoritmo en scala que calcule el radio de un circulo

var pi: Double = 3.1416;
var circuferencia: Double = 15;
var radio: Double =  (circuferencia / (2*pi));

//2. Desarrollar un algoritmo en scala que me diga si un numero es primo
var a: Int = 2

var b: Int = 1
if(a/a == 1)
{
 println("Es primo");
}
   

//3. Dada la variable bird = "tweet", utiliza interpolacion de string para
//   imprimir "Estoy ecribiendo un tweet"
val bird = "tweet";
printf("Estoy escribiendo un %s" , bird);
//4. Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la
//   secuencia "Luke"

val darth = "Hola Luke yo soy tu padre!"
val result = darth.slice(5,10)
println("[" + result + "]")

//5. Cual es la diferencia en value y una variable en scala?
//  R.- cuando declaramos un variable con val esta no puede ser modificada (es inmutable) mientras que con var podemos modificarla (mutable)
//6. Dada la tupla ((2,4,5),(1,2,3),(3.1416,23))) regresa el numero 3.1416 


val my_tup = ((2,4,5),(1,2,3),(3.1416,23))
(3,1,(2,3))

my_tup._3
my_tup._1





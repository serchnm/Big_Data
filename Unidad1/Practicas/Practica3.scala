//Versión recursiva descendente
def fib1(n: Int): Int = {
  n match {
    case i if i < 2 => i
    case i => fib_recur(n-1) + fib_recur(n-2)
  }
}
fib1(6)

//Versión con fórmula explícita

def fibonacciFormula(num: Int): Double = {
    if(num < 2){
        return num; 
    }else{
        var i = ((1 + math.sqrt(5))/2);
        var j = ((math.pow(i,num)) - (math.pow((1-i),num)))/ math.sqrt(5);
        return j;
    }
}

//Versión iterativa

def fibonacciIterativa(num: Int): Int = {
    var a = 0;
    var b = 1;
    var c = 0;
    var k = 1;
    for(k <- 1 to num){
        c = b + a;
        a = b;
        b = c;
    }
    return a;
}

//Versión iterativa 2 variable
def fibonacciIterativa2(num: Int): Int = {
    var a = 0;
    var b = 1;
    for(k <- 1 to num){
        b = b + a;
        a = b - a;
    }
    return a;
}

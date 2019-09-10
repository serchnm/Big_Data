def fib1(n: Int): Int = {
  n match {
    case i if i < 2 => i
    case i => fib_recur(n-1) + fib_recur(n-2)
  }
}
fib1(6)
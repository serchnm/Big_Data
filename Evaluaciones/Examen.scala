
var totalGames = 9
var game = Array(10,5,20,20,4,5,2,25,1)
var game2 = Array(3,4,21,36,10,28,35,5,24,42) 

def breakingRecords( game:Array[Int]): List[Int] = {
    
    var max = game(0)
    var countMax = 0
    var min = game(0)
    var countMin = 0
    for (i <- game){

        if(i > max){
            max = i 
            countMax += 1
        }else if(i < min){
            min = i
            countMin += 1
        }
    }
    var myList = List(countMax,countMin)
    return myList
}

println(breakingRecords(game))
println(breakingRecords(game2))













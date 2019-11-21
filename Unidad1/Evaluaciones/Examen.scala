
var game = Array(10,5,20,20,4,5,2,25,1)
var game2 = Array(3,4,21,36,10,28,35,5,24,42) 

def breakingRecords( game:Array[Int]): List[Int] = {
    
    var maxGames = game(0)
    var countMaxGames = 0
    var minGames = game(0)
    var countMinGames = 0
    for (x <- game){

        if(x > maxGames){
            maxGames = x 
            countMaxGames += 1
        }else if(x < minGames){
            minGames = x
            countMinGames += 1
        }
    }
    var myList = List(countMaxGames,countMinGames)
    return myList
}
println("----------------")
println("Sample Input 0")
var Input0 = 9
println("----------------")

println(breakingRecords(game))

println("----------------")
println("Sample Input 1")
var Input1 =10

println("----------------")
println(breakingRecords(game2))













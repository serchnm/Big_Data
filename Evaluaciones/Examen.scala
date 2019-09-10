
var Input0 = 9
var Input1 =10
var game = Array(10,5,20,20,4,5,2,25,1)
var game2 = Array(3,4,21,36,10,28,35,5,24,42) 

def breakingRecords( game:Array[Int]): List[Int] = {
    
    var maxG = game(0)
    var countMaxG = 0
    var minG = game(0)
    var countMinG = 0
    for (i <- game){

        if(i > maxG){
            maxG = i 
            countMaxG += 1
        }else if(i < minG){
            minG = i
            countMinG += 1
        }
    }
    var myList = List(countMaxG,countMinG)
    return myList
}

println(breakingRecords(game))
println(breakingRecords(game2))













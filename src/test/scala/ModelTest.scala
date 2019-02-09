import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.FunSuite

//class ModelTest extends FunSuite{
//
//  test("keras forward "){
//
//        val outSize = 30
//        val skuCount = 30
//
//        val kerasRNN = new SessionRecommender()
//        val model1 = kerasRNN.buildModel(outSize, skuCount, 10, 20)
//
//         val feature = Tensor(Array(2.0f,4,6,8,10,12,14,16,18,20),Array(1,10))
//
//         val output = model1.forward(feature)
//
//         println(output)
//
//
//  }
//
//  test("bigdl forward "){
//
//    val outSize = 30
//    val skuCount = 30
//
//    val rnn = new BigDLRNN()
//    val model2 = rnn.buildModel(outSize, skuCount, 10, 20)
//
//    val feature = Tensor(Array(2.0f,4,6,8,10,12,14,16,18,20),Array(1,10))
//
//    val output = model2.forward(feature)
//
//    println(output)
//
//
//  }
//}

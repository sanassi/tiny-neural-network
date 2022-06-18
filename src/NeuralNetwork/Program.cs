using System;

namespace NeuralNetwork
{
    class Program
    {
        static void SolveXor(int nbHidden, double trainingEta, int nbTrainingEpoch)
        {
            // test nn on xor problem : appears to be working
            
            int nbInput = 2;
            int nbHidden = 50;
            int nbOutput = 1;
            

            int inputDataSetLength = 4;

            NeuralNetwork net = new NeuralNetwork(nbInput, nbHidden, nbOutput, 0.5);
            
            // push 2d array data in jagged array (to be fixed..) (more convinient bc i can pass entire array as parameter)
            double[,] inputData2D = { { 0.0, 0.0 }, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
            double[][] inputData = new double[inputDataSetLength][];

            for (int i = 0; i < inputData.GetLength(0); i++)
            {
                inputData[i] = new double[nbInput];
                inputData[i][0] = inputData2D[i, 0];
                inputData[i][1] = inputData2D[i, 1];
            }
            //

            double[] targetData = { 0, 1, 1, 0};

            // create training order
            int[] order = new int[4];

            for (int i = 0; i < 4; i++)
            {
                order[i] = i;
            }



            for (int epoch = 0; epoch < 5000; epoch++)
            {
                Random rnd = new Random();
                
                // randomize training order foreach epoch
                rnd.Shuffle(order);
                
                for (int i = 0; i < 4; i++)
                {
                    net.Train(inputData[order[i]], new double[] {targetData[order[i]]});
                }
            }
            
            
            // display results

            for (int i = 0; i < 4; i++)
            {
                net.FeedForward(new double[] {inputData[i][0], inputData[i][1]});
                Console.WriteLine($"{inputData[i][0]} xor {inputData[i][1]} = {Math.Round(net.output._matrix[0, 0])}");
            }
        }
        
        static void Main(string[] args)
        {
            SolveXor(50, 0.5, 5000);
        }
    }
}

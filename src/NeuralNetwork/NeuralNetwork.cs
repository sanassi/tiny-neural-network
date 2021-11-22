using System;
using System.IO;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        public int nbInput;
        public int nbHidden;
        public int nbOutput;

        public Matrix input;
        public Matrix hidden;
        public Matrix output;

        public Matrix w_i_h;
        public Matrix w_h_o;

        public Matrix b_h;
        public Matrix b_o;

        public double eta;


        // ***********************Constructors****************************
        public NeuralNetwork(int nbInput, int nbHidden, int nbOutput, double eta)
        {
            this.nbInput = nbInput;
            this.nbHidden = nbHidden;
            this.nbOutput = nbOutput;

            this.input = new Matrix(nbInput, 1);
            this.hidden = new Matrix(nbHidden, 1);
            this.output = new Matrix(nbOutput, 1);

            this.w_i_h = new Matrix(nbHidden, nbInput, 0, 1);
            this.w_h_o = new Matrix(nbOutput, nbHidden, 0, 1);

            this.b_h = new Matrix(nbHidden, 1, 0, 1);
            this.b_o = new Matrix(nbOutput, 1, 0, 1);

            this.eta = eta;


            /*
            for (int i = 0; i < nbHidden; i++)
            {
                for (int j = 0; j < nbInput; j++)
                {
                    this.w_i_h[i, j] = 0;
                }

                this.hidden[i] = 0;
                this.b_h[i] = GetRandomNumber(0, 1);
            }

            for (int i = 0; i < nbOutput; i++)
            {
                for (int j = 0; j < nbHidden; j++)
                {
                    this.w_h_o[i, j] = 0;
                }

                this.output[i] = 0;
                this.b_o[i] = GetRandomNumber(0, 1);
            }
            */
        }
        
        public NeuralNetwork(string networkPath) // init network with previously saved network dats
        {
            using (StreamReader sr = new StreamReader(networkPath + "netInfo.txt"))
            {
                this.nbInput = Convert.ToInt32(sr.ReadLine());
                this.nbHidden = Convert.ToInt32(sr.ReadLine());
                this.nbOutput = Convert.ToInt32(sr.ReadLine());

                this.eta = Convert.ToDouble(sr.ReadLine());
            }
            
            this.input = new Matrix(nbInput, 1);
            this.hidden = new Matrix(nbHidden, 1);
            this.output = new Matrix(nbOutput, 1);

            this.w_i_h = new Matrix(nbHidden, nbInput, 0, 1);
            this.w_h_o = new Matrix(nbOutput, nbHidden, 0, 1);

            this.b_h = new Matrix(nbHidden, 1, 0, 1);
            this.b_o = new Matrix(nbOutput, 1, 0, 1);
            
            this.Load(networkPath);
        }
        
        //****************************************************************
        
        
        public void FeedForward(double[] inputArr)
        {
            /*
            for (int i = 0; i < nbHidden; i++)
            {
                double sum = 0;
                
                for (int j = 0; j < nbInput; j++)
                {
                    sum += (w_i_h[i, j]) * (input[j]);
                }

                sum += b_h[i];
                hidden[i] = Sigmoid(sum);
            }
            */

            input = Matrix.FromArray(inputArr, inputArr.Length, 1);
            this.hidden = Matrix.Multiply(w_i_h, input);
            hidden = Matrix.Add(hidden, b_h);

            hidden = Matrix.Map(hidden, x => Sigmoid(x));

            /*
            for (int i = 0; i < nbOutput; i++)
            {
                double sum = 0;
                
                for (int j = 0; j < nbHidden; j++)
                {
                    sum += (w_h_o[i, j]) * (hidden[j]);
                }

                sum += b_o[i];

                output[i] = Sigmoid(sum);
            }
            */

            output = Matrix.Multiply(w_h_o, hidden);
            output = Matrix.Add(output, b_o);
            output = Matrix.Map(output, x => Sigmoid(x));
        }
        public void Train(double[] inputArr, double[] targetArr)
        {
            FeedForward(inputArr);

            Matrix targets = Matrix.FromArray(targetArr, targetArr.Length, 1);
            
            Matrix outputError = Matrix.Subtract(targets, output);
            
            /*
            output.Print();
            Console.WriteLine();
            targets.Print();
            Console.WriteLine();
            error.Print();
            */

            Matrix gradients = Matrix.Map(output, d => dSigmoid(d));
            
            //output = Matrix.Map(output, x => dSigmoid(x));
            gradients = Matrix.HadamardProduct(gradients, outputError);
            gradients = Matrix.Multiply(gradients, this.eta);

            Matrix hiddenTranspose = Matrix.Transpose(hidden);
            Matrix weightsHoDelta = Matrix.Multiply(gradients, hiddenTranspose);

            // adjust the weights by deltas
            this.w_h_o = Matrix.Add(this.w_h_o, weightsHoDelta);
            
            // adjust the bias by its delta (gradients)
            this.b_o = Matrix.Add(b_o, gradients);
            

            // hidden layer errors
            Matrix weightsHiddenOutputTranspose = Matrix.Transpose(this.w_h_o);
            Matrix hiddenError = Matrix.Multiply(weightsHiddenOutputTranspose, outputError);

            Matrix hiddenGradient = Matrix.Map(hidden, d => dSigmoid(d));
            hiddenGradient = Matrix.HadamardProduct(hiddenGradient, hiddenError);
            hiddenGradient = Matrix.Multiply(hiddenGradient, this.eta);
            
            // calculate input -> hidden deltas

            Matrix inputsTranspose = Matrix.Transpose(this.input);
            Matrix weightsInputHiddenDeltas = Matrix.Multiply(hiddenGradient, inputsTranspose);

            this.w_i_h = Matrix.Add(this.w_i_h, weightsInputHiddenDeltas);
            this.b_h = Matrix.Add(b_h, hiddenGradient);

        }
        
        // Save and Load NN

        public void Save(string folderPath)
        {
            using (StreamWriter sr =
                new StreamWriter(folderPath + "netInfo.txt"))
            {
                sr.WriteLine(nbInput);
                sr.WriteLine(nbHidden);
                sr.WriteLine(nbOutput);
                
                sr.WriteLine(eta);
            }
            
            
            using (StreamWriter sw =
                new StreamWriter(folderPath + "w_i_h.txt"))
            {
                for (int i = 0; i < nbHidden; i++)
                {
                    for (int j = 0; j < nbInput; j++)
                    {
                        sw.WriteLine(w_i_h._matrix[i, j]);
                    }
                }
            }
            
            using (StreamWriter sw =
                new StreamWriter(folderPath + "w_h_o.txt"))
            {
                for (int i = 0; i < nbOutput; i++)
                {
                    for (int j = 0; j < nbHidden; j++)
                    {
                        sw.WriteLine(w_h_o._matrix[i, j]);
                    }
                }
            }
            
            using (StreamWriter sw =
                new StreamWriter(folderPath + "b_h.txt"))
            {
                for (int i = 0; i < nbHidden; i++)
                {
                    sw.WriteLine(b_h._matrix[i, 0]);
                    
                }
            }
            
            using (StreamWriter sw =
                new StreamWriter(folderPath + "b_o.txt"))
            {
                for (int i = 0; i < nbOutput; i++)
                {
                    sw.WriteLine(b_o._matrix[i, 0]);
                    
                }
            }
        }
        
        public static double ConvertToDouble(string Value) {
            if (Value == null) {
                return 0;
            }
            else {
                double OutVal;
                double.TryParse(Value, out OutVal);

                if (double.IsNaN(OutVal) || double.IsInfinity(OutVal)) {
                    return 0;
                }
                return OutVal;
            }
        }

        public void Load(string folderPath)
        {
            using (StreamReader sr = 
                new StreamReader(folderPath + "w_i_h.txt"))
            {
                for (int i = 0; i < nbHidden; i++)
                {
                    for (int j = 0; j < nbInput; j++)
                    {
                        w_i_h._matrix[i, j] = Convert.ToDouble(sr.ReadLine());
                    }
                }
            }
            
            using (StreamReader sr = 
                new StreamReader(folderPath + "w_h_o.txt"))
            {
                for (int i = 0; i < nbOutput; i++)
                {
                    for (int j = 0; j < nbHidden; j++)
                    {
                        w_h_o._matrix[i, j] = Convert.ToDouble(sr.ReadLine());
                    }
                }
            }
            
            using (StreamReader sr = 
                new StreamReader(folderPath + "b_h.txt"))
            {
                for (int i = 0; i < nbHidden; i++)
                {
                    b_h._matrix[i, 0] = Convert.ToDouble(sr.ReadLine());
                }
            }
            
            using (StreamReader sr = 
                new StreamReader(folderPath + "b_o.txt"))
            {
                for (int i = 0; i < nbOutput; i++)
                {
                    b_o._matrix[i, 0] = Convert.ToDouble(sr.ReadLine());
                }
            }
        }


        // tests
        
        public void Print()
        {
            for (int i = 0; i < nbInput; i++)
            {
                Console.Write(input._matrix[i, 0] + " ");
            }

            Console.WriteLine();
            
            for (int i = 0; i < nbHidden; i++)
            {
                Console.Write(hidden._matrix[i, 0] + " ");
            }
            
            Console.WriteLine();
            
            for (int i = 0; i < nbOutput; i++)
            {
                Console.Write(output._matrix[i, 0] + " ");
            }
        }

        public int GetMaxOutputActivationIndex()
        {
            int res = 0;

            for (int i = 0; i < this.nbOutput; i++)
            {
                if (this.output._matrix[i, 0] >= this.output._matrix[res, 0])
                    res = i;
            }

            return res;
        }


        // misc 
        public double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public double dSigmoid(double x)
        {
            //return Sigmoid(x) * (1 - Sigmoid(x));
            return x * (1 - x);
        }

        public double GetRandomNumber(double minimum, double maximum)
        { 
            Random random = new Random();
            return random.NextDouble() * (maximum - minimum) + minimum;
        }
        
        public static double[] GetTargetArray(int arrSize, double target, double targetValue)
        {
            double[] tmp = new double[arrSize];

            for (int j = 0; j < arrSize; j++)
                tmp[j] = 0;

            tmp[(int) target] = targetValue;

            return tmp;
        }
    }
}
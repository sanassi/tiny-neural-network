using System;

namespace NeuralNetwork
{
    public class Matrix
    {
        // matrix class to make matrix operations easier 
        private int _row;
        private int _col;
        public double[,] _matrix;

        // todo : change functions to non static 
        
       /*****************************Constructors******************************/
        public Matrix(int row, int col)
        {
            this._row = row;
            this._col = col;

            this._matrix = new double[_row, _col];
        }
        
        public Matrix(int row, int col, double initVal)
        {
            this._row = row;
            this._col = col;

            this._matrix = new double[_row, _col];

            for (int i = 0; i < _row; i++)
            {
                for (int j = 0; j < _col; j++)
                {
                    this._matrix[i, j] = initVal;
                }
            }
        }

        public Matrix(int row, int col, double min = Double.MinValue, double max = Double.MaxValue) // create matrix and set values between min and max
        {
            this._row = row;
            this._col = col;

            this._matrix = new double[_row, _col];
            
            Random random = new Random();

            for (int i = 0; i < _row; i++)
            {
                for (int j = 0; j < _col; j++)
                {
                    this._matrix[i, j] = random.NextDouble() * (max - min) + min; // generate random double between min and max
                }
            }
        }
        /************************************************************************/
        
        /****************************Matrix Operations***************************/
        public static Matrix Add(Matrix A, Matrix B)
        {
            if (A._row != B._row || A._col != B._col)
            {
                throw new ArgumentException("Error : Add Matrix Invalid Dimensions");
            }
            
            Matrix res = new Matrix(A._row, A._col);

            for (int i = 0; i < res._row; i++)
            {
                for (int j = 0; j < res._col; j++)
                {
                    res._matrix[i, j] = A._matrix[i, j] + B._matrix[i, j];
                }
            }

            return res;
        }
        
        public static Matrix Add(Matrix A, double x)
        {
            Matrix res = new Matrix(A._row, A._col);

            for (int i = 0; i < res._row; i++)
            {
                for (int j = 0; j < res._col; j++)
                {
                    res._matrix[i, j] = A._matrix[i, j] + x;
                }
            }

            return res;
        }
        
        public static Matrix Subtract(Matrix A, Matrix B)
        {
            if (A._row != B._row || A._col != B._col)
            {
                throw new ArgumentException("Error : Add Matrix Invalid Dimensions");
            }
            
            Matrix res = new Matrix(A._row, A._col);

            for (int i = 0; i < res._row; i++)
            {
                for (int j = 0; j < res._col; j++)
                {
                    res._matrix[i, j] = A._matrix[i, j] - B._matrix[i, j];
                }
            }

            return res;
        }
        
        public static Matrix Subtract(Matrix A, double x)
        {
            Matrix res = new Matrix(A._row, A._col);

            for (int i = 0; i < res._row; i++)
            {
                for (int j = 0; j < res._col; j++)
                {
                    res._matrix[i, j] = A._matrix[i, j] - x;
                }
            }

            return res;
        }

        public static Matrix Multiply(Matrix A, double alpha)
        {
            Matrix res = new Matrix(A._row, A._col);

            for (int i = 0; i < A._row; i++)
            {
                for (int j = 0; j < res._col; j++)
                {
                    res._matrix[i, j] = A._matrix[i, j] * alpha;
                }
            }

            return res;
        }

        public static Matrix Multiply(Matrix A, Matrix B)
        {
            if (A._col != B._row)
            {
                throw new ArgumentException("Error : Multiply Matrix Invalid Dimensions");
            }
            
            Matrix res = new Matrix(A._row, B._col);

            for (int i = 0; i < res._row; i++)
            {
                for (int j = 0; j < res._col; j++)
                {
                    double sum = 0;

                    for (int k = 0; k < B._row; k++)
                    {
                        sum += A._matrix[i, k] * B._matrix[k, j];
                    }

                    res._matrix[i, j] = sum;
                }
            }

            return res;
        }

        public static Matrix HadamardProduct(Matrix A, Matrix B) // element-wise vector multiplication
        {
            Matrix res = new Matrix(A._row, A._col);

            for (int i = 0; i < res._row; i++)
            {
                for (int j = 0; j < res._col; j++)
                {
                    res._matrix[i, j] = A._matrix[i, j] * B._matrix[i, j];
                }
            }

            return res;
        }

        public static Matrix Map(Matrix A, Func<double, double> func)
        {
            Matrix res = new Matrix(A._row, A._col);

            for (int i = 0; i < res._row; i++)
            {
                for (int j = 0; j < res._col; j++)
                {
                    res._matrix[i, j] = func(A._matrix[i, j]);
                }
            }

            return res;
        }

        public static Matrix Transpose(Matrix A)
        {
            Matrix res = new Matrix(A._col, A._row);

            for (int i = 0; i < res._row; i++)
            {
                for (int j = 0; j < res._col; j++)
                {
                    res._matrix[i, j] = A._matrix[j, i];
                }
            }

            return res;
        }
        
        public static Matrix FromArray(double[] inputArr, int row, int col) // turn array into matrix given row an col info
        {
            Matrix M = new Matrix(row, col);
            
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    M._matrix[i, j] = inputArr[i * col + j];
                }
            }

            return M;
        }

        public void Print()
        {
            for (int i = 0; i < _row; i++)
            {
                for (int j = 0; j < _col; j++)
                {
                    Console.Write(_matrix[i, j] + " ");
                }
                
                Console.WriteLine();
            }
        }
    }
}
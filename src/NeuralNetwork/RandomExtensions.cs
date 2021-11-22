using System;

namespace NeuralNetwork
{
    public static class RandomExtensions
    {
        public static void Shuffle<T> (this Random rng, T[] array) // randomize array 
        {
            int n = array.Length;
            while (n > 1) 
            {
                int k = rng.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }
    }
}
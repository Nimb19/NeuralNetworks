using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Neuron
    {
        public List<double> Weights { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }

        public Neuron(int inputCount, NeuronType neuronType = NeuronType.Normal)
        {
            NeuronType = neuronType;
            Weights = new List<double>();

            for (int i = 0; i < inputCount; i++)
            {
                Weights.Add(1);
            }
        }

        public double FeedForward(List<double> inputs)
        {
            double sum = 0.0;

            for (int i = 0; i < inputs.Count; i++)
            {
                sum = inputs[i] * Weights[i];
            }

            Output = Sigmoid(sum);

            return Output;
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public void SetWeights(params double[] weights)
        {
            // TODO: Удалить после добавления возможности обучения сети.
            for (int i = 0; i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}

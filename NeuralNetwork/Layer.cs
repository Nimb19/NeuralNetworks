using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }
        public int Count => Neurons?.Count ?? 0;
        
        public Layer(List<Neuron> neurons, NeuronType neuronType = NeuronType.Normal)
        {
            DataValidation(neurons, neuronType);

            Neurons = neurons;
        }

        private void DataValidation(List<Neuron> neurons, NeuronType neuronType)
        {
            if (neurons.Count == 0)
            {
                throw new ArgumentException("Количество нейронов должно быть больше 0.", nameof(neurons));
            }
            NeuronType neuronType1 = neurons[0].NeuronType;

            for (int i = 0; i < neurons.Count; i++)
            {
                if (neurons[i].NeuronType != neuronType1)
                {
                    throw new ArgumentException("Все нейроны в списке должны быть одного типа NeuronType.", nameof(neurons));
                }
            }
        }

        public List<double> GetSignals()
        {
            var result = new List<double>();

            foreach (var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }

            return result;
        }
    }
}

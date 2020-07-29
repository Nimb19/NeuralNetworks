using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        public List<Layer> Layers { get; }
        public Topology Topology { get; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayer();
            CreateOutputLayers();
        }

        public Neuron FeedForward(List<double> inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if (Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(x => x.Output).First();
            }
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayerSignals = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }
            }
        }

        private void SendSignalsToInputNeurons(List<double> inputSignals)
        {
            for (int i = 0; i < inputSignals.Count; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }

        private void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();

            for (int i = 0; i < Topology.InputCount; i++)
            {
                Neuron neuron = new Neuron(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }

            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }

        private void CreateHiddenLayer()
        {
            for (int c = 0; c < Topology.HiddenLayers.Count; c++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayerCount = Layers.Last().Count;

                for (int i = 0; i < Topology.HiddenLayers[c]; i++)
                {
                    Neuron neuron = new Neuron(lastLayerCount, NeuronType.Normal);
                    hiddenNeurons.Add(neuron);
                }

                var hiddenLayer = new Layer(hiddenNeurons, NeuronType.Normal);
                Layers.Add(hiddenLayer);
            }
        }

        private void CreateOutputLayers()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayerCount = Layers.Last().Count;

            for (int i = 0; i < Topology.OutputCount; i++)
            {
                Neuron neuron = new Neuron(lastLayerCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }

            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }
    }
}

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <chrono>

// Define Neural Network structure and operations (simplified)
struct NeuralNetwork {
    int inputSize;
    int hiddenSize;
    int outputSize;

    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<double> biasHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;
    std::vector<double> biasOutput;

    NeuralNetwork(int input, int hidden, int output)
        : inputSize(input), hiddenSize(hidden), outputSize(output) {
        // Initialize weights and biases randomly
        initializeWeightsAndBiases();
    }

    double sigmoid(double x) const {
        return 1 / (1 + exp(-x));
    }

    std::vector<double> softmax(const std::vector<double>& input) const {
        std::vector<double> result;
        double sumExp = 0.0;
        for (double val : input) {
            sumExp += exp(val);
        }
        for (double val : input) {
            result.push_back(exp(val) / sumExp);
        }
        return result;
    }

    void initializeWeightsAndBiases() {
        // Initialize weights and biases for input to hidden layer
        weightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize));
        #pragma omp parallel for
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < hiddenSize; ++j) {
                weightsInputHidden[i][j] = getRandomWeight();
            }
        }
        biasHidden.resize(hiddenSize);
        #pragma omp parallel for
        for (int i = 0; i < hiddenSize; ++i) {
            biasHidden[i] = getRandomWeight();
        }

        // Initialize weights and biases for hidden to output layer
        weightsHiddenOutput.resize(hiddenSize, std::vector<double>(outputSize));
        #pragma omp parallel for
        for (int i = 0; i < hiddenSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                weightsHiddenOutput[i][j] = getRandomWeight();
            }
        }
        biasOutput.resize(outputSize);
        #pragma omp parallel for
        for (int i = 0; i < outputSize; ++i) {
            biasOutput[i] = getRandomWeight();
        }
    }

    double getRandomWeight() const {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<double> dis(-1.0, 1.0);
        return dis(gen);
    }

    std::vector<double> forward(const std::vector<double>& input) const {
        std::vector<double> hidden(hiddenSize);
        #pragma omp parallel for
        for (int i = 0; i < hiddenSize; ++i) {
            double sum = 0.0;
            for (int j = 0; j < inputSize; ++j) {
                sum += input[j] * weightsInputHidden[j][i];
            }
            hidden[i] = sigmoid(sum + biasHidden[i]);
        }

        std::vector<double> output(outputSize);
        #pragma omp parallel for
        for (int i = 0; i < outputSize; ++i) {
            double sum = 0.0;
            for (int j = 0; j < hiddenSize; ++j) {
                sum += hidden[j] * weightsHiddenOutput[j][i];
            }
            output[i] = sum + biasOutput[i];
        }

        return softmax(output);
    }
      ~NeuralNetwork() {
    // Destructor to properly deallocate memory
    for (auto& row : weightsInputHidden) {
      row.clear();
    }
    weightsInputHidden.clear();
    biasHidden.clear();
    for (auto& row : weightsHiddenOutput) {
      row.clear();
    }
    weightsHiddenOutput.clear();
    biasOutput.clear();
  }

};

std::pair<std::vector<std::vector<double>>, std::vector<int>> generateDummyData(int numSamples, int inputSize, int numClasses, std::mt19937& gen) {
    std::normal_distribution<> normalDist(0.0, 1.0);

    std::vector<std::vector<double>> data;
    std::vector<int> labels;

    for (int i = 0; i < numSamples; ++i) {
        std::vector<double> sample(inputSize, 0.0);
        // Generate features with a more complex distribution
        for (int j = 0; j < inputSize; ++j) {
            // Use a combination of normal distributions and non-linear transformations
            double x = normalDist(gen);
            double y = normalDist(gen);
            sample[j] = sin(x) * cos(y); // Example of non-linear transformation
        }
        data.push_back(sample);
        labels.push_back(gen() % numClasses); // Randomly assign labels
    }

    return {data, labels};
}

// Define a Fitness function to evaluate the performance of a neural network
double evaluateFitness(const NeuralNetwork& network, const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    // Evaluate the fitness of the neural network based on some criteria
    // e.g., accuracy, error rate, etc.
    int correctPredictions = 0;
    int totalSamples = data.size();

    #pragma omp parallel for reduction(+:correctPredictions)
    for (int i = 0; i < totalSamples; ++i) {
        std::vector<double> output = network.forward(data[i]);
        int predictedClass = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        if (predictedClass == labels[i]) {
            correctPredictions++;
        }
    }

    return static_cast<double>(correctPredictions) / totalSamples; // Accuracy
}

// Define a Chromosome representing a neural network's parameters
struct Chromosome {
    NeuralNetwork network;
    double fitness;
};

// Define a Genetic Algorithm class
class GeneticAlgorithm {
private:
    std::vector<Chromosome> population;
    int populationSize;
    double mutationRate;
    int inputSize;
    int hiddenSize;
    int outputSize;
    std::vector<std::vector<double>> data;
    std::vector<int> labels;

    // Function to initialize the population with random neural networks
    void initializePopulation() {
        for (int i = 0; i < populationSize; ++i) {
            NeuralNetwork network(inputSize, hiddenSize, outputSize);
            population.push_back({network, 0.0});
        }
    }

    // Function to evaluate the fitness of each chromosome in the population
    void evaluatePopulation() {
        #pragma omp parallel for
        for (auto& chromosome : population) {
            chromosome.fitness = evaluateFitness(chromosome.network, data, labels);
        }
    }

    // Function to select parents based on fitness (Tournament Selection)
    Chromosome selectParent() {
        std::shuffle(population.begin(), population.end(), std::default_random_engine());
        return (population[0].fitness > population[1].fitness) ? population[0] : population[1];
    }

    // Function to perform crossover (Single Point Crossover)
    Chromosome crossover(const Chromosome& parent1, const Chromosome& parent2) {
        NeuralNetwork childNetwork(inputSize, hiddenSize, outputSize);
        // Implement crossover operation to combine parameters of parent networks
        return {childNetwork, 0.0};
    }

    // Function to perform mutation (Random Mutation)
    void mutate(Chromosome& chromosome) {
        // Perform random mutation on the chromosome's neural network parameters
        // Mutation can involve changing weights, biases, etc.
    }

public:
    GeneticAlgorithm(int popSize, double mutationRate, int input, int hidden, int output)
        : populationSize(popSize), mutationRate(mutationRate), inputSize(input), hiddenSize(hidden), outputSize(output) {
        initializePopulation();
    }

    void setData(const std::vector<std::vector<double>>& inputData, const std::vector<int>& inputLabels) {
        data = inputData;
        labels = inputLabels;
    }

    // Function to evolve the population
    void evolve(int generation) {
        evaluatePopulation();

        std::vector<Chromosome> newPopulation;

        // Elitism: Keep the best chromosome from the previous generation
        auto bestChromosome = std::max_element(population.begin(), population.end(),
                                               [](const Chromosome& a, const Chromosome& b) {
                                                   return a.fitness < b.fitness;
                                               });
        newPopulation.push_back(*bestChromosome);

        // Generate offspring until the new population is full
        #pragma omp parallel for
        for (int i = 1; i < populationSize; ++i) {
            // Select parents
            Chromosome parent1 = selectParent();
            Chromosome parent2 = selectParent();

            // Perform crossover to produce offspring
            Chromosome offspring = crossover(parent1, parent2);

            // Mutate offspring
            mutate(offspring);

            // Add offspring to the new population
            #pragma omp critical
            newPopulation.push_back(offspring);
        }

        // Update population
        population = std::move(newPopulation);

        // Calculate and print the average fitness of the population
        double totalFitness = 0.0;
        for (const auto& chromosome : population) {
            totalFitness += chromosome.fitness;
        }
        double averageFitness = totalFitness / populationSize;

        // Write the average fitness to a file
        std::cout << "Generation " << generation << ": Average Fitness = " << averageFitness << std::endl;

        std::ofstream outputFile("generation_parallel_fitness.txt", std::ios::app);
        if (outputFile.is_open()) {
            outputFile << "Generation " << generation << ": Average Fitness = " << averageFitness << std::endl;
            outputFile.close();
        } else {
            std::cerr << "Error: Unable to open output file." << std::endl;
        }
    }
};

int main() {

    std::random_device rd;
    std::mt19937 gen(rd());

    auto startTime = std::chrono::high_resolution_clock::now();
    int populationSize = 500;
    double mutationRate = 0.3;
    int inputSize = 10; // Example input size
    int hiddenSize = 20; // Example hidden layer size
    int outputSize = 5; // Example output size
    GeneticAlgorithm ga(populationSize, mutationRate, inputSize, hiddenSize, outputSize);

    // Generate dummy data for classification
    int numSamples = 2000;
    int numClasses = outputSize;
    auto [data, labels] = generateDummyData(numSamples, inputSize, numClasses, gen);
    ga.setData(data, labels);

    // Evolve for a certain number of generations
    int numGenerations = 1000;
    for (int i = 0; i < numGenerations; ++i) {
        ga.evolve(i + 1); // Pass current generation number
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> totalTime = endTime - startTime;

    // Print total time taken
    std::cout << "Total time taken: " << totalTime.count() << " seconds" << std::endl;

    return 0;
}

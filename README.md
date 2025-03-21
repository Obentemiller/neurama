# neurama
A lightweight and efficient implementation of neural networks in C++.


## Features

- **Layered Architecture**: Supports multi-layer networks with customizable input and output sizes.
- **Activation Functions**: Provides ReLU and Sigmoid activations, with easy integration of additional functions.
- **Weight Initialization**: Offers random and Xavier initialization methods for setting initial weights.
- **Optimizers**: Includes Stochastic Gradient Descent (SGD) and Adam optimizers for training.
- **Loss Function**: Utilizes Mean Squared Error (MSE) for measuring prediction accuracy.

## Getting Started

To incorporate Neurama into your project:

1. **Include the Header**: Add `#include "neurama.h"` to your source file.
2. **Create a Neural Network**: Instantiate a `NeuralNetwork` object.
3. **Add Layers**: Use the `addLayer` method to define layers, specifying input size, output size, activation function, and initialization method.
4. **Train the Network**: Provide training data and call the `trainModel` function, specifying the number of epochs and learning rate.
5. **Make Predictions**: Use the `forward` method to obtain outputs for new inputs.

## Example

```cpp
#include "neurama.h"
#include <vector>
#include <iostream>

int main() {
    // 🚀 Inicialização da Rede Neural
    NeuralNetwork nn;
    nn.addLayer(2, 4, actMethod::ReLU, InitMethod::XAVIER);
    nn.addLayer(4, 1, actMethod::sigmoid, InitMethod::XAVIER);

    // 🔍 Dados de treinamento para o problema XOR
    std::vector<std::vector<double>> trainInputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    std::vector<std::vector<double>> trainTargets = { {0}, {1}, {1}, {0} };

    // 💡 Exemplo de entrada para visualização do treinamento
    std::vector<double> sampleInput = {0, 1};

    // ⚙️ Treinamento da rede
    std::cout << "Iniciando treinamento... 🔥" << std::endl;
    nn.trainModel(trainInputs, trainTargets, 1000, 0.01, sampleInput, OptimizerType::ADAM);
    std::cout << "Treinamento concluído! ✅" << std::endl;

    // 🧪 Testando a rede com os dados de treinamento
    std::cout << "\nResultados dos testes:" << std::endl;
    nn.testModel(trainInputs, trainTargets);

    return 0;
}

```

## Documentation

For detailed information on Neurama's classes, functions, and customization options, refer to the [Neurama Documentation](https://github.com/Obentemiller/neurama).

## Contributions

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Commit your changes.
4. Submit a pull request detailing your modifications.

## License

Neurama is licensed under the mozilla license. See the [LICENSE](LICENSE) file for more information.

## Acknowledgments

Neurama draws inspiration from various open-source neural network libraries, including:

- [MiniDNN](https://github.com/yixuan/MiniDNN): A header-only C++ library for deep neural networks.
- [OpenNN](https://www.opennn.net/): An open-source neural networks library for machine learning.
- [FANN](https://github.com/libfann/fann): Fast Artificial Neural Network Library.
- [tensorflow](https://github.com/tensorflow/tensorflow): The most powerful framework today.

These projects have influenced Neurama's design and functionality. 

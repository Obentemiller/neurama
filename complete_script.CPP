#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <numeric> // Para std::inner_product

using namespace std;

// Tipos para ponteiros de funções de ativação e sua derivada
typedef double (*ActivationFunc)(double);
typedef double (*ActivationFuncDeriv)(double);

struct ActivationPair {
    ActivationFunc func;
    ActivationFuncDeriv deriv;
};

// Funções de ativação e respectivas derivadas
inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

inline double sigmoid_deriv(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

inline double ReLU(double x) {
    return (x > 0) ? x : 0;
}

inline double ReLU_deriv(double x) {
    return (x > 0) ? 1 : 0;
}

// Enum para escolher o método de inicialização dos pesos
enum class InitMethod {
    RANDOM, // Valores aleatórios em [-0.5, 0.5]
    XAVIER  // Inicialização Xavier
};

enum class actMethod {
    ReLU,    // Ativação ReLU
    sigmoid  // Ativação Sigmoid
};

ActivationPair activationMatrix[] = {
    { ReLU, ReLU_deriv },       // Para actMethod::ReLU
    { sigmoid, sigmoid_deriv }  // Para actMethod::sigmoid
};

// Enum para escolha do otimizador
enum class OptimizerType {
    SGD,
    ADAM
};

// Classe que representa uma camada totalmente conectada
class Layer {
public:
    int input_size;
    int output_size;
    vector<vector<double>> weights; // Matriz de pesos [output_size x input_size]
    vector<double> biases;          // Vieses [output_size]
    
    // Valores intermediários para forward e backward (pré-alocados)
    vector<double> input;
    vector<double> z_values;
    vector<double> output;
    vector<double> delta;
    
    // Ponteiros para a função de ativação e sua derivada
    ActivationFunc activation;
    ActivationFuncDeriv activation_deriv;
    
    // Parâmetros para o otimizador Adam
    vector<vector<double>> m_weights, v_weights;
    vector<double> m_biases, v_biases;
    int adam_t;
    
    // Constantes para o Adam
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double epsilon = 1e-8;
    
    // Construtor: recebe tamanhos, funções de ativação e método de inicialização
    Layer(int in_size, int out_size, ActivationFunc act, ActivationFuncDeriv act_deriv, InitMethod init_method = InitMethod::RANDOM)
    : input_size(in_size), output_size(out_size), activation(act), activation_deriv(act_deriv), adam_t(0)
    {
        weights.resize(output_size, vector<double>(input_size));
        biases.resize(output_size);
        
        // Pré-aloca os vetores de forward e backward
        z_values.resize(output_size);
        output.resize(output_size);
        delta.resize(output_size);
        
        // Inicializa arrays do Adam com zeros
        m_weights.resize(output_size, vector<double>(input_size, 0.0));
        v_weights.resize(output_size, vector<double>(input_size, 0.0));
        m_biases.resize(output_size, 0.0);
        v_biases.resize(output_size, 0.0);
        
        if (init_method == InitMethod::RANDOM) {
            // Inicialização padrão: valores aleatórios em [-0.5, 0.5]
            for (int i = 0; i < output_size; i++) {
                for (int j = 0; j < input_size; j++) {
                    weights[i][j] = ((double) rand() / RAND_MAX) - 0.5;
                }
                biases[i] = ((double) rand() / RAND_MAX) - 0.5;
            }
        } else if (init_method == InitMethod::XAVIER) {
            // Inicialização Xavier: intervalo baseado em sqrt(6/(fan_in+fan_out))
            double limit = sqrt(6.0 / (input_size + output_size));
            for (int i = 0; i < output_size; i++) {
                for (int j = 0; j < input_size; j++) {
                    weights[i][j] = ((double) rand() / RAND_MAX) * 2 * limit - limit;
                }
                biases[i] = ((double) rand() / RAND_MAX) * 2 * limit - limit;
            }
        }
    }
    
    // Forward pass: calcula z = w*x + b e aplica a função de ativação definida
    vector<double> forward(const vector<double>& in) {
        input = in; // Armazena o input atual
        // Utiliza std::inner_product para acelerar a soma ponderada
        for (int i = 0; i < output_size; i++) {
            double z = biases[i] + inner_product(weights[i].begin(), weights[i].end(), in.begin(), 0.0);
            z_values[i] = z;
            output[i] = activation(z);
        }
        return output;
    }
    
    // Backward pass: calcula gradientes, atualiza pesos e vieses, e retorna gradiente para camada anterior
    vector<double> backward(const vector<double>& grad_output, double learning_rate, OptimizerType opt = OptimizerType::SGD) {
        // Calcula delta para cada neurônio
        for (int i = 0; i < output_size; i++) {
            delta[i] = grad_output[i] * activation_deriv(z_values[i]);
        }
        
        // Calcula o gradiente para a camada anterior
        vector<double> grad_input(input_size, 0.0);
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                grad_input[j] += weights[i][j] * delta[i];
            }
        }
        
        if (opt == OptimizerType::SGD) {
            // Atualização via gradiente descendente simples (SGD)
            for (int i = 0; i < output_size; i++) {
                for (int j = 0; j < input_size; j++) {
                    weights[i][j] -= learning_rate * delta[i] * input[j];
                }
                biases[i] -= learning_rate * delta[i];
            }
        } else if (opt == OptimizerType::ADAM)  {
            // Atualização via otimizador Adam
            adam_t++;
            double beta1_t = pow(beta1, adam_t);
            double beta2_t = pow(beta2, adam_t);
            
            for (int i = 0; i < output_size; i++) {
                for (int j = 0; j < input_size; j++) {
                    double grad = delta[i] * input[j];
                    m_weights[i][j] = beta1 * m_weights[i][j] + (1 - beta1) * grad;
                    v_weights[i][j] = beta2 * v_weights[i][j] + (1 - beta2) * grad * grad;
                    double m_hat = m_weights[i][j] / (1 - beta1_t);
                    double v_hat = v_weights[i][j] / (1 - beta2_t);
                    weights[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
                }
                double grad_bias = delta[i];
                m_biases[i] = beta1 * m_biases[i] + (1 - beta1) * grad_bias;
                v_biases[i] = beta2 * v_biases[i] + (1 - beta2) * grad_bias * grad_bias;
                double m_hat_bias = m_biases[i] / (1 - beta1_t);
                double v_hat_bias = v_biases[i] / (1 - beta2_t);
                biases[i] -= learning_rate * m_hat_bias / (sqrt(v_hat_bias) + epsilon);
            }
        }
        return grad_input;
    }
};

class NeuralNetwork {
public:
    vector<Layer*> layers;
    
    NeuralNetwork() {}
    
    ~NeuralNetwork() {
        for (auto layer : layers)
            delete layer;
    }
    
    // Adiciona uma camada com as especificações do usuário
    void addLayer(int in_size, int out_size, actMethod method, InitMethod init_method = InitMethod::RANDOM) {
        int idx = static_cast<int>(method);
        ActivationFunc act = activationMatrix[idx].func;
        ActivationFuncDeriv act_deriv = activationMatrix[idx].deriv;
        layers.push_back(new Layer(in_size, out_size, act, act_deriv, init_method));
    }
    
    // Propagação direta por todas as camadas
    vector<double> forward(const vector<double>& input) {
        vector<double> out = input;
        for (auto layer : layers) {
            out = layer->forward(out);
        }
        return out;
    }
    
    // Backpropagation: atualiza os pesos de todas as camadas com o otimizador escolhido
    void backward(const vector<double>& loss_grad, double learning_rate, OptimizerType opt = OptimizerType::SGD) {
        vector<double> grad = loss_grad;
        for (int i = layers.size() - 1; i >= 0; i--) {
            grad = layers[i]->backward(grad, learning_rate, opt);
        }
    }
};

// Função de perda: Erro Quadrático Médio (MSE)
double mse_loss(const vector<double>& predicted, const vector<double>& target) {
    double sum = 0.0;
    for (size_t i = 0; i < predicted.size(); i++) {
        double diff = predicted[i] - target[i];
        sum += diff * diff;
    }
    return sum / predicted.size();
}

// Gradiente do MSE em relação à saída
vector<double> mse_loss_grad(const vector<double>& predicted, const vector<double>& target) {
    vector<double> grad(predicted.size());
    for (size_t i = 0; i < predicted.size(); i++) {
        grad[i] = 2 * (predicted[i] - target[i]) / predicted.size();
    }
    return grad;
}

// Função de treinamento: realiza o treinamento e retorna a predição para um exemplo (sampleInput)
vector<double> trainModel(NeuralNetwork &nn, 
                          const vector<vector<double>> &trainInputs, 
                          const vector<vector<double>> &trainTargets, 
                          int epochs, 
                          double learning_rate, 
                          const vector<double>& sampleInput,
                          OptimizerType opt = OptimizerType::SGD) {
    for (int epoch = 0; epoch < epochs; epoch++){
        double epoch_loss = 0.0;
        for (size_t i = 0; i < trainInputs.size(); i++) {
            vector<double> output = nn.forward(trainInputs[i]);
            double loss = mse_loss(output, trainTargets[i]);
            epoch_loss += loss;
            vector<double> loss_grad = mse_loss_grad(output, trainTargets[i]);
            nn.backward(loss_grad, learning_rate, opt);
        }
        if (epoch % 1000 == 0){
            cout << "Epoch " << epoch 
                 << " - Loss Média: " << epoch_loss / trainInputs.size() 
                 << endl;
        }
    }
    return nn.forward(sampleInput);
}

// Função de teste: avalia o modelo com os dados de teste
void testModel(NeuralNetwork &nn, 
               const vector<vector<double>> &testInputs, 
               const vector<vector<double>> &testTargets) {
    cout << "\nResultados do Teste:" << endl;
    for (size_t i = 0; i < testInputs.size(); i++){
        vector<double> prediction = nn.forward(testInputs[i]);
        cout << "Input: (";
        for (size_t j = 0; j < testInputs[i].size(); j++) {
            cout << testInputs[i][j] << (j < testInputs[i].size() - 1 ? ", " : "");
        }
        cout << ") -> Predito: ";
        for (double p : prediction)
            cout << p << " ";
        cout << " | Alvo: ";
        for (double t : testTargets[i])
            cout << t << " ";
        cout << endl;
    }
}

int main() {
    srand(static_cast<unsigned>(time(0)));
    
    // Exemplo de criação e treinamento de uma rede neural simples
    NeuralNetwork nn;
    nn.addLayer(2, 4, actMethod::ReLU, InitMethod::XAVIER);
    nn.addLayer(4, 1, actMethod::sigmoid, InitMethod::XAVIER);
    
    // Dados de treinamento (exemplo simples)
    vector<vector<double>> trainInputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    vector<vector<double>> trainTargets = { {0}, {1}, {1}, {0} };
    
    // Treinamento
    vector<double> sampleInput = {0, 1};
    vector<double> prediction = trainModel(nn, trainInputs, trainTargets, 1000, 0.01, sampleInput, OptimizerType::ADAM);
    
    cout << "\nPredição para sampleInput: ";
    for (double p : prediction)
        cout << p << " ";
    cout << endl;
    
    // Teste
    testModel(nn, trainInputs, trainTargets);
    
    return 0;
}

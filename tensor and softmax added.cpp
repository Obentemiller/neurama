#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <cmath>

using namespace std;

template<typename T>
class Tensor {
public:
    Tensor(vector<size_t> dimensions) : dimensions_(dimensions) {
        size_t totalSize = 1;
        for (size_t dim : dimensions) {
            totalSize *= dim;
        }
        try {
            data_.resize(totalSize);
        } catch (const bad_alloc& e) {
            cerr << "Erro de alocação de memória: " << e.what() << endl;
            throw;
        }
    }
    
    T& operator[](const vector<size_t>& indices) {
        size_t linearIndex = getLinearIndex(indices);
        return data_[linearIndex];
    }
    
    const T& operator[](const vector<size_t>& indices) const {
        size_t linearIndex = getLinearIndex(indices);
        return data_[linearIndex];
    }
    
    void print() const {
        size_t totalElements = 1;
        for (size_t dim : dimensions_) {
            totalElements *= dim;
        }
        for (size_t idx = 0; idx < totalElements; ++idx) {
            cout << data_[idx] << " ";
            if ((idx + 1) % dimensions_[0] == 0) {
                cout << endl;
            }
        }
    }
    
private:
    vector<size_t> dimensions_;
    vector<T> data_;
    
    size_t getLinearIndex(const vector<size_t>& indices) const {
        size_t linearIndex = 0;
        size_t stride = 1;
        for (size_t i = indices.size(); i-- > 0;) {
            linearIndex += indices[i] * stride;
            stride *= dimensions_[i];
        }
        return linearIndex;
    }
};

typedef double (*ActivationFunc)(double);
typedef double (*ActivationFuncDeriv)(double);

struct ActivationPair {
    ActivationFunc func;
    ActivationFuncDeriv deriv;
};


inline double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

inline double sigmoid_deriv(double x){
    double s = sigmoid(x);
    return s * (1 - s);
}

inline double ReLU(double x){
    return (x > 0) ? x : 0;
}

inline double ReLU_deriv(double x){
    return (x > 0) ? 1 : 0;
}

inline double tanh_activation(double x){
    return (exp(x)-exp(-x))/(exp(x)+exp(-x));
}

inline double tanh_deriv(double x){
    return 1 - (tanh_activation(x) * tanh_activation(x));
}

inline double leakyReLU(double x) {
    return (x > 0) ? x : 0.01 * x;
}

inline double leakyReLU_deriv(double x) {
    return (x > 0) ? 1 : 0.01;
}

inline double elu(double x) {
    return (x >= 0) ? x : (exp(x) - 1);
}

inline double elu_deriv(double x) {
    return (x >= 0) ? 1 : exp(x);
}

inline double softplus(double x) {
    return log(1 + exp(x));
}

inline double softplus_deriv(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// NOVO: Função softmax – implementada como função auxiliar que opera sobre um vetor (era a unica forma de fazer isso sem mudar tudo)...
vector<double> softmax_vector(const vector<double>& z) {
    vector<double> result(z.size());
    double max_z = *max_element(z.begin(), z.end());
    double sum = 0.0;
    for (double val : z) {
        sum += exp(val - max_z);
    }
    for (size_t i = 0; i < z.size(); i++) {
        result[i] = exp(z[i] - max_z) / sum;
    }
    return result;
}

inline double softmax_activation(double x) {
    return x;
}

inline double softmax_deriv(double x) {
    return 1.0;
}


ActivationPair activationMatrix[] = {
    {ReLU, ReLU_deriv},
    {sigmoid, sigmoid_deriv},
    {tanh_activation, tanh_deriv},
    {leakyReLU, leakyReLU_deriv},
    {elu, elu_deriv},
    {softplus, softplus_deriv},
    {softmax_activation, softmax_deriv} 
};

enum class InitMethod {
    RANDOM, // Valores aleatórios em [-0.5, 0.5]
    XAVIER,
    tanh    // Inicialização Xavier
};

enum class actMethod {
    ReLU,      // 0
    sigmoid,   // 1
    tanh,      // 2
    leakyReLU, // 3
    elu,       // 4
    softplus,  // 5
    softmax    // 6
};

enum class OptimizerType {
    SGD,
    ADAM
};

// ================================================
// Novas definições para funções de erro (loss)
// ================================================
typedef double (*LossFunc)(const vector<double>&, const vector<double>&);
typedef vector<double> (*LossFuncDeriv)(const vector<double>&, const vector<double>&);

struct LossPair {
    LossFunc loss;
    LossFuncDeriv deriv;
};

enum class LossMethod {
    MSE,               // Erro Quadrático Médio
    MAE,               // Erro Absoluto Médio
    BinaryCrossEntropy // Entropia Cruzada Binária
};

// --------------------
// Função MSE e derivada
// --------------------
inline double mse_loss(const vector<double>& predicted, const vector<double>& target) {
    double sum = 0.0;
    for (size_t i = 0; i < predicted.size(); i++) {
        double diff = predicted[i] - target[i];
        sum += diff * diff;
    }
    return sum / predicted.size();
}

inline vector<double> mse_loss_grad(const vector<double>& predicted, const vector<double>& target) {
    vector<double> grad(predicted.size());
    for (size_t i = 0; i < predicted.size(); i++) {
        grad[i] = 2 * (predicted[i] - target[i]) / predicted.size();
    }
    return grad;
}

// --------------------
// Função MAE e derivada
// --------------------
inline double mae_loss(const vector<double>& predicted, const vector<double>& target) {
    double sum = 0.0;
    for (size_t i = 0; i < predicted.size(); i++) {
        sum += fabs(predicted[i] - target[i]);
    }
    return sum / predicted.size();
}

inline vector<double> mae_loss_grad(const vector<double>& predicted, const vector<double>& target) {
    vector<double> grad(predicted.size());
    for (size_t i = 0; i < predicted.size(); i++) {
        double diff = predicted[i] - target[i];
        if(diff > 0)
            grad[i] = 1.0 / predicted.size();
        else if(diff < 0)
            grad[i] = -1.0 / predicted.size();
        else
            grad[i] = 0.0;
    }
    return grad;
}

// -------------------------------
// Função Binary Cross Entropy e derivada
// -------------------------------
inline double binary_cross_entropy_loss(const vector<double>& predicted, const vector<double>& target) {
    double sum = 0.0;
    double eps = 1e-12;
    for (size_t i = 0; i < predicted.size(); i++) {
        sum += -(target[i] * log(predicted[i] + eps) + (1 - target[i]) * log(1 - predicted[i] + eps));
    }
    return sum / predicted.size();
}

inline vector<double> binary_cross_entropy_grad(const vector<double>& predicted, const vector<double>& target) {
    vector<double> grad(predicted.size());
    double eps = 1e-12;
    for (size_t i = 0; i < predicted.size(); i++) {
        grad[i] = (-(target[i] / (predicted[i] + eps)) + ((1 - target[i]) / (1 - predicted[i] + eps))) / predicted.size();
    }
    return grad;
}

// Atualizando a matriz de funções de erro para incluir as novas funções
LossPair lossMatrix[] = {
    { mse_loss, mse_loss_grad },
    { mae_loss, mae_loss_grad },
    { binary_cross_entropy_loss, binary_cross_entropy_grad }
};

// =====================================================
// Camada totalmente conectada utilizando Tensores
// =====================================================
class Layer {
public:
    int input_size;
    int output_size;
    Tensor<double> weights; // Tensor de dimensões [output_size x input_size]
    Tensor<double> biases;  // Tensor de dimensões [output_size]
    
    // Vetores auxiliares para forward e backward (mantidos como 1D)
    vector<double> input;
    vector<double> z_values;
    vector<double> output;
    vector<double> delta;
    
    ActivationFunc activation;
    ActivationFuncDeriv activation_deriv;
    
    // Flag para identificar se esta camada utiliza softmax
    bool use_softmax;
    
    // Tensores para os parâmetros do otimizador Adam
    Tensor<double> m_weights, v_weights;
    Tensor<double> m_biases, v_biases;
    int adam_t;
    
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double epsilon = 1e-8;
    
    // Construtor: inicializa os tensores e realiza a inicialização dos pesos
    Layer(int in_size, int out_size, ActivationFunc act, ActivationFuncDeriv act_deriv, InitMethod init_method = InitMethod::RANDOM)
    : input_size(in_size), output_size(out_size),
      weights({(size_t)out_size, (size_t)in_size}),
      biases({(size_t)out_size}),
      m_weights({(size_t)out_size, (size_t)in_size}),
      v_weights({(size_t)out_size, (size_t)in_size}),
      m_biases({(size_t)out_size}),
      v_biases({(size_t)out_size}),
      activation(act), activation_deriv(act_deriv), adam_t(0)
    {
        // Determina se a camada usará softmax (comparando com nossa função dummy)
        use_softmax = (activation == softmax_activation);
        
        // Pré-alocação dos vetores para operações forward e backward
        z_values.resize(output_size);
        output.resize(output_size);
        delta.resize(output_size);
        
        // Inicialização dos pesos e vieses
        if (init_method == InitMethod::RANDOM) {
            for (int i = 0; i < output_size; i++) {
                for (int j = 0; j < input_size; j++) {
                    weights[{(size_t)i, (size_t)j}] = ((double) rand() / RAND_MAX) - 0.5;
                }
                biases[{(size_t)i}] = ((double) rand() / RAND_MAX) - 0.5;
            }
        } else if (init_method == InitMethod::XAVIER) {
            double limit = sqrt(6.0 / (input_size + output_size));
            for (int i = 0; i < output_size; i++) {
                for (int j = 0; j < input_size; j++) {
                    weights[{(size_t)i, (size_t)j}] = ((double) rand() / RAND_MAX) * 2 * limit - limit;
                }
                biases[{(size_t)i}] = ((double) rand() / RAND_MAX) * 2 * limit - limit;
            }
        }
        
        // Inicializa os tensores do Adam com zero
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                m_weights[{(size_t)i, (size_t)j}] = 0.0;
                v_weights[{(size_t)i, (size_t)j}] = 0.0;
            }
            m_biases[{(size_t)i}] = 0.0;
            v_biases[{(size_t)i}] = 0.0;
        }
    }
    
    // Forward pass: calcula z = w*x + b e aplica a função de ativação
    vector<double> forward(const vector<double>& in) {
        input = in;
        for (int i = 0; i < output_size; i++) {
            double z = biases[{(size_t)i}];
            for (int j = 0; j < input_size; j++) {
                z += weights[{(size_t)i, (size_t)j}] * in[j];
            }
            z_values[i] = z;
        }
        if (use_softmax) {
            // Para softmax, aplicamos a função no vetor completo
            output = softmax_vector(z_values);
        } else {
            for (int i = 0; i < output_size; i++) {
                output[i] = activation(z_values[i]);
            }
        }
        return output;
    }
    
    // Backward pass: calcula gradientes, atualiza pesos/vieses e retorna o gradiente para a camada anterior
    vector<double> backward(const vector<double>& grad_output, double learning_rate, OptimizerType opt = OptimizerType::SGD) {
        if (use_softmax) {
            // Para softmax (normalmente usado com cross-entropy), a derivada já é incorporada no loss_grad.
            for (int i = 0; i < output_size; i++) {
                delta[i] = grad_output[i];
            }
        } else {
            for (int i = 0; i < output_size; i++) {
                delta[i] = grad_output[i] * activation_deriv(z_values[i]);
            }
        }
        
        vector<double> grad_input(input_size, 0.0);
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                grad_input[j] += weights[{(size_t)i, (size_t)j}] * delta[i];
            }
        }
        
        if (opt == OptimizerType::SGD) {
            for (int i = 0; i < output_size; i++) {
                for (int j = 0; j < input_size; j++) {
                    weights[{(size_t)i, (size_t)j}] -= learning_rate * delta[i] * input[j];
                }
                biases[{(size_t)i}] -= learning_rate * delta[i];
            }
        } else if (opt == OptimizerType::ADAM)  {
            adam_t++;
            double beta1_t = pow(beta1, adam_t);
            double beta2_t = pow(beta2, adam_t);
            
            for (int i = 0; i < output_size; i++) {
                for (int j = 0; j < input_size; j++) {
                    double grad = delta[i] * input[j];
                    m_weights[{(size_t)i, (size_t)j}] = beta1 * m_weights[{(size_t)i, (size_t)j}] + (1 - beta1) * grad;
                    v_weights[{(size_t)i, (size_t)j}] = beta2 * v_weights[{(size_t)i, (size_t)j}] + (1 - beta2) * grad * grad;
                    double m_hat = m_weights[{(size_t)i, (size_t)j}] / (1 - beta1_t);
                    double v_hat = v_weights[{(size_t)i, (size_t)j}] / (1 - beta2_t);
                    weights[{(size_t)i, (size_t)j}] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
                }
                double grad_bias = delta[i];
                m_biases[{(size_t)i}] = beta1 * m_biases[{(size_t)i}] + (1 - beta1) * grad_bias;
                v_biases[{(size_t)i}] = beta2 * v_biases[{(size_t)i}] + (1 - beta2) * grad_bias * grad_bias;
                double m_hat_bias = m_biases[{(size_t)i}] / (1 - beta1_t);
                double v_hat_bias = v_biases[{(size_t)i}] / (1 - beta2_t);
                biases[{(size_t)i}] -= learning_rate * m_hat_bias / (sqrt(v_hat_bias) + epsilon);
            }
        }
        return grad_input;
    }
};

// =====================================================
// Classe NeuralNetwork
// =====================================================
class NeuralNetwork {
public:
    vector<Layer*> layers;
    
    NeuralNetwork() {}
    
    ~NeuralNetwork() {
        for (auto layer : layers)
            delete layer;
    }
    
    // Adiciona uma camada especificando tamanho, método de ativação e inicialização
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
    
    // Backpropagation: atualiza os pesos de todas as camadas usando o otimizador escolhido
    void backward(const vector<double>& loss_grad, double learning_rate, OptimizerType opt = OptimizerType::SGD) {
        vector<double> grad = loss_grad;
        for (int i = layers.size() - 1; i >= 0; i--) {
            grad = layers[i]->backward(grad, learning_rate, opt);
        }
    }
};

// =====================================================
// Função de treinamento utilizando a estrutura de loss
// =====================================================
vector<double> trainModel(NeuralNetwork &nn, 
                          const vector<vector<double>> &trainInputs, 
                          const vector<vector<double>> &trainTargets, 
                          int epochs, 
                          double learning_rate, 
                          const vector<double>& sampleInput,
                          OptimizerType opt = OptimizerType::SGD,
                          LossMethod loss_method = LossMethod::MSE) {
    int loss_idx = static_cast<int>(loss_method);
    for (int epoch = 0; epoch < epochs; epoch++){
        double epoch_loss = 0.0;
        for (size_t i = 0; i < trainInputs.size(); i++) {
            vector<double> output = nn.forward(trainInputs[i]);
            double loss = lossMatrix[loss_idx].loss(output, trainTargets[i]);
            epoch_loss += loss;
            vector<double> loss_grad = lossMatrix[loss_idx].deriv(output, trainTargets[i]);
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

// =====================================================
// Função de teste: avalia o modelo com os dados de teste
// =====================================================
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

/*
funções de ativação:

    ReLU,      // 0
    sigmoid,   // 1
    tanh,      // 2
    leakyReLU, // 3
    elu,       // 4
    softplus   // 5

funções de otimização:

    SGD,
    ADAM

funções de erro:

    MSE,               // Erro Quadrático Médio
    MAE,               // Erro Absoluto Médio
    BinaryCrossEntropy // Entropia Cruzada Binária
*/



int main() {
    // Exemplo de uso (exemplo para o XOR)
    srand(time(NULL));
    
    NeuralNetwork nn;
    nn.addLayer(2, 250, actMethod::tanh, InitMethod::XAVIER);
    nn.addLayer(250, 1, actMethod::sigmoid, InitMethod::XAVIER);
    
    // Dados de treino para o XOR
    vector<vector<double>> trainInputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    vector<vector<double>> trainTargets = {
        {0},
        {1},
        {1},
        {0}
    };
    
    vector<double> sampleInput = {1, 0};
    vector<double> prediction = trainModel(nn, trainInputs, trainTargets, 100,
         0.1, sampleInput, OptimizerType::ADAM, LossMethod::BinaryCrossEntropy);
    
    cout << "\nPredição para input (1, 0): ";
    for (double p : prediction)
        cout << p << " ";
    cout << endl;
    
    testModel(nn, trainInputs, trainTargets);
    
    return 0;
} 

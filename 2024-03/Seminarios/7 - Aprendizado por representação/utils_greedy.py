# -- coding: utf-8 --
"""utils.py
    """
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


def prepare_data(x, y):
    """
        Função para preparar os dados. Cria dados sintéticos usando make_blobs para gerar clusters de pontos em duas
        dimensões, com três centros de clusters. Os dados são convertidos em uma forma de codificação one-hot e,
        em seguida, divididos em conjuntos de treino e teste

        :return: train_x, test_x, train_y, test_y
        """
    # Codificar a saída em one-hot
    y = to_categorical(y)
    # Dividir em conjunto de treino e teste
    n_train = 500
    train_x, test_x = x[:n_train, :], x[n_train:, :]
    train_y, test_y = y[:n_train], y[n_train:]

    return train_x, test_x, train_y, test_y


# define and fit the base model
def get_base_model(train_x, trainy):
    """

    :param train_x:
    :param trainy:
    :return:
    """
    # define model
    model = Sequential()
    model.add(Dense(10, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(3, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # fit model
    model.fit(train_x, trainy, epochs=100, verbose=0)
    return model


# evaluate a fit model
def evaluate_model(model, train_x, test_x, trainy, testy):
    """

    :param model:
    :param train_x:
    :param test_x:
    :param trainy:
    :param testy:
    :return:
    """
    train_loss, train_acc = model.evaluate(train_x, trainy, verbose=0)
    test_loss, test_acc = model.evaluate(test_x, testy, verbose=0)
    
    return train_loss, test_loss, train_acc, test_acc


# add one new layer and re-train only the new layer
def add_layer(model, train_x, trainy):
    """

    :param model:
    :param train_x:
    :param trainy:
    """
    # remember the current output layer
    output_layer = model.layers[-1]
    # remove the output layer
    model.pop()
    # mark all remaining layers as non-trainable
    for layer in model.layers:
        layer.trainable = False
    # add a new hidden layer
    model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
    # re-add the output layer
    model.add(output_layer)
    # fit model
    model.fit(train_x, trainy, epochs=100, verbose=0)


def base_autoencoder(train_x, test_x):
    """
        :param train_x:
        :param test_x:

        Função para definir, treinar e avaliar o autoencoder base.
        Cria um modelo de autoencoder básico.
        Estrutura: uma camada de entrada de 2 dimensões, uma camada oculta de 10 neurônios com ativação ReLU
        e uma camada de saída linear para reconstruir os dados de entrada.
        Compilação: o modelo é compilado com perda de erro quadrático médio (mse) e o otimizador SGD.
        Avaliação: o autoencoder é treinado para reconstruir as entradas e calcula o erro de reconstrução
        nos conjuntos de treino e teste.

        :return: model_1
        """
    # Definir o modelo
    model_1 = Sequential()
    n_neuron = 10
    set_dim = 2
    model_1.add(Dense(n_neuron, input_dim=set_dim, activation='relu', kernel_initializer='he_uniform'))
    model_1.add(Dense(2, activation='linear'))
    # Compilar o modelo
    model_1.compile(loss='mse', optimizer=SGD(learning_rate=0.01, momentum=0.9))
    # Treinar o modelo
    model_1.fit(train_x, train_x, epochs=100, verbose=0)
    # Avaliar o erro de reconstrução
    train_mse = model_1.evaluate(train_x, train_x, verbose=0)
    test_mse = model_1.evaluate(test_x, test_x, verbose=0)

    print('> reconstruction error train=%.3f, test=%.3f' % (train_mse, test_mse))

    return model_1


def evaluate_autoencoder_as_classifier(model_eval, train_x, train_y, test_x, test_y):
    """
        :param model_eval:
        :param train_x:
        :param train_y:
        :param test_x:
        :param test_y:
        Avalia a qualidade do autoencoder como um classificador.
        Primeiramente, remove a última camada do autoencoder (camada de saída) e congela as demais camadas
        (deixa essas camadas não treináveis).
        Em seguida, uma nova camada de saída softmax é adicionada, permitindo que o modelo realize uma classificação
        multi-classe.
        O modelo é treinado para classificação e avaliado em termos de precisão nos conjuntos de treino e teste.
        Ao final, a camada de saída de softmax é removida, e a camada original é re-adicionada para manter a
        estrutura inicial do autoencoder.

        :return: train_acc_model, test_acc_model
        """
    # Remover a última camada (output)
    output_layer = model_eval.layers[-1]
    model_eval.pop()
    # Congelar as camadas restantes
    for layer in model_eval.layers:
        layer.trainable = False
    # Adicionar nova camada de saída para classificação
    model_eval.add(Dense(3, activation='softmax'))
    # Compilar o modelo com função de perda para classificação
    model_eval.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01, momentum=0.9),
                       metrics=['accuracy'])
    # Treinar o modelo
    model_eval.fit(train_x, train_y, epochs=100, verbose=0)
    # Avaliar o modelo
    train_loss_model, train_acc_model = model_eval.evaluate(train_x, train_y, verbose=0)
    test_loss_model, test_acc_model = model_eval.evaluate(test_x, test_y, verbose=0)
    # Restaurar o modelo para autoencoder
    model_eval.pop()
    model_eval.add(output_layer)
    model_eval.compile(loss='mse', optimizer=SGD(learning_rate=0.01, momentum=0.9))

    return train_loss_model, test_loss_model, train_acc_model, test_acc_model


def add_layer_to_autoencoder(model_eval, train_x, test_x):
    """
        :param model_eval:
        :param train_x:
        :param test_x:

        Adiciona uma nova camada oculta de 10 neurônios ao modelo (inicializada com ativação ReLU).
        Congela as camadas anteriores para manter os parâmetros previamente aprendidos e impede que sejam modificados.
        Treina apenas a nova camada adicionada, mantendo a estrutura de autoencoder
        e calcula novamente o erro de reconstrução nos conjuntos de treino e teste.

        :return: train_acc_model, test_acc_model
        """
    # Remover a última camada e congelar as camadas existentes
    output_layer = model_eval.layers[-1]
    model_eval.pop()
    for layer in model_eval.layers:
        layer.trainable = False
    # Adicionar nova camada oculta
    model_eval.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
    # Re-adicionar a camada de saída original
    model_eval.add(output_layer)
    # Treinar o modelo com a nova camada
    model_eval.fit(train_x, train_x, epochs=100, verbose=0)
    # Avaliar o erro de reconstrução
    train_mse = model_eval.evaluate(train_x, train_x, verbose=0)
    test_mse = model_eval.evaluate(test_x, test_x, verbose=0)
    print('> reconstruction error train=%.3f, test=%.3f' % (train_mse, test_mse))

"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random #Importa una libreria que permite tener numeros aleatorios entre muchas otras funcionalidades

# Third-party libraries
import numpy as np #Libreria que permite la introducción de matrices entre otras muchas funciones

class Network(object): #Declaración de nuestra clase, desciende de la clase objeto mediante herencia

    def __init__(self, sizes): #Declaración del constructor. Al instanciar un objeto de tipo network, se le debe pasar una lista
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)#Atributo de la clase, se le asigna el número de elementos de la lista que se pasa al instanciar un objeto de la clase
        self.sizes = sizes #Atributo de la clase, se le asigna al atributo la lista que se pasa al instanciar la clase
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]#Creación de un atributo, el cual es una lista con los bais. Es generado apartir de números aleatorios. Es una lista de matrices, cada matriz tendrá "y" filas y 1 columna.
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])] #Atributo de la clase el cuál es igual a una lista de matrices, las cuales tendrá "y" filas y "x" columnas. x, y se obtienen de sizes, x toma todos los valores excepto el de la capa de salida, y toma todos excepto el de la capa de entrada. Los elementos de todas las matrices son numeros entre -1 y 1

    def feedforward(self, a): #Se define un método de la clase. Toma como parámetro la activación
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):#El zip "empaqueta" los elementos de los atributos biases y weights de forma 1 a 1 y le asigna dichos valores a "b" y "w"
            a = sigmoid(np.dot(w, a)+b) #Le asigna a "a" el valor de la sigmoide realizando el producto punto de la matriz de pesos y de activación, y le añade el valor del bias
        return a #Regresa el valor de "a"

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None): #Definición de un atributo de la clase
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: #Si se proporcionó el parámetro test_data al llamar a la función
            test_data = list(test_data)#Se convierte a test data en una lista
            n_test = len(test_data)#Se guarda el número de elementos en una variable

        training_data = list(training_data)#Se convierte a lista
        n = len(training_data)#Se guarda el número de elementos de la lista
        for j in range(epochs): #Se comienza a iterar acorde al número de épocas o ciclos
            random.shuffle(training_data) #Mezcla de manera aleatoria los elementos de la lista training_data
            mini_batches = [#Creación de la lista donde irán los minibatches para entrenar la red
                training_data[k:k+mini_batch_size] #Se toma el elemento k y los "mini_batch_size"(tamaño del minibatch) elementos siguientes. Esto se hace varias veces pues se está iterando, desde 0 hasta n dando brincos acorde al tamaño del minibatch deseado.Como resultado se tiene listas de listas
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:#Se iteran los elementos de la lista minibatches
                self.update_mini_batch(mini_batch, eta)#Se manda a llamar un método definido más abajo pasandole como parámetros un minibatch y la eta. Profe, olvidé el nombre que tiene este valor pero si sé que controla "el tamaño de cada paso" cuando intentamos minimizar la función de costo
            if test_data: #Si se proporcionó test_data
                print("Epoch {0}: {1} / {2}".format( #Imprime en la consola "Epoch (valor de j): (valor del método evaulate)/(número de pruebas o entrenamientos)
                    j, self.evaluate(test_data), n_test))
            else: #Si no: 
                print("Epoch {0} complete".format(j)) #Imprime en consola: "Epoca (valor de j) complete"

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

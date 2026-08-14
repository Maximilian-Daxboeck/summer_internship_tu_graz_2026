import numpy as np


# n Features [x1, x2, x3, ... xn]
# k Klassen [y1, y2, ..., yk]

# f: Features -> Klassen, f(x) = argmax_y P(y|x) proportional zu P(x|y)P(y)
# wobei:
#   * P(y)... Prior = Wahrscheinlichkeit, mit der die Klasse y allgemein auftritt
#   * P(x|y)=P(x1, x2, ..., x_n|y) (wegen naiver Annahme der Unabhängigkeit der einzelnen Features) = P(x1|y) P(x2|y) ... P(xn|y) ... Likelihood

# Prior und Likelihood werden für die gegebenen Daten (alle möglichen Kombinationen) bestimmt. -> Dann Modell fertig

# P(y) = Anzahl aller y zugeordneten Datenpunkte / Gesamtanzahl
# P(xk|y) = Anzahl aller Punkte mit xk -> y / Anzahl aller y zugeordneten Datenpunkte

# 'Anzahl aller y zugeordneten Datenpunkte' kommt 2 Mal vor, muss aber nur einmal berechnet werden

# -> Ergebnis: TENSOR :( mit Likelihood und k-dim. Vektor mit den Priors der einzelnen Klassen


def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x)

def softmax_strong(x):
    a = np.max(x)
    return softmax(x-a)

def normalize(x):
    return x / np.sum(x)


def get_model(features, labels, possible_feature_values): # -> Likelihood-Tensor, Prior-Vektor
    number_of_possible_feature_values = len(possible_feature_values)

    # Prior vector:
    N = features.shape[0]
    number_of_features = features.shape[1]

    l_T = labels.transpose(1,0)

    number_of_classes = labels.shape[1]
    prior_sum = np.zeros(number_of_classes)
    for i in range(number_of_classes):
        prior_sum[i] = sum(l_T[i])


    prior = prior_sum / N
    # - --~*~-~*~-~*~-~*~-~*~-~*~-~*~-~*~-~*~-- -
    # Likelihood tensor:

    likelihood_matrix_list = [np.zeros((number_of_features, number_of_possible_feature_values)) for i in range(number_of_classes)]

    for i in range(N):
        class_index = np.argmax(labels[i])
        for j in range(number_of_features):
            type_index = possible_feature_values.index(features[i][j])
            likelihood_matrix_list[class_index][j][type_index] += 1

    for i in range(number_of_classes):
        likelihood_matrix_list[i] = likelihood_matrix_list[i] / prior_sum[i]

    likelihood_tensor = np.array(likelihood_matrix_list)


    return likelihood_tensor, prior



def map_classify(x, possible_feature_values, likelihood, prior):
    number_of_classes = likelihood.shape[0]
    number_of_features = likelihood.shape[1]
    number_of_possible_feature_values = likelihood.shape[2]

    posterior_of_classes = np.zeros(number_of_classes)
    for i in range(number_of_classes):
        class_likelihoods = likelihood[i]
        feature_likelihoods = []
        for j in range(number_of_features):
            feature_type_index = possible_feature_values.index(x[j])
            feature_likelihoods += [class_likelihoods[j][feature_type_index]]

        posterior = np.prod(feature_likelihoods) * prior[i]

        posterior_of_classes[i] = posterior


    distribution = normalize(posterior_of_classes)
    predicted_class = np.argmax(posterior_of_classes)
    return predicted_class, distribution

if __name__ == "__main__":
    features = np.array([[1,0,0],
                         [0,1,0],
                         [0,1,1],
                         [1,0,1]])

    possible_feature_values = [0,1]

    labels = np.array([[1,0],
                       [1,0],
                       [0,1],
                       [0,1]])


    l, p = get_model(features, labels, possible_feature_values)

    print(l, p)

    print(map_classify(np.array([0,0,1]), possible_feature_values, l,p)[0])

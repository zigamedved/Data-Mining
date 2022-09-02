import math
import re
import matplotlib.pyplot as plt
import numpy as np

from unidecode import unidecode
from os import listdir
from os.path import join
from collections import Counter
from sklearn.manifold import MDS, TSNE


def terke(text, n):
    """
    Vrne slovar s preštetimi terkami dolžine n.
    """
    text = unidecode(text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)

    res = []
    for i in range(len(text) - n + 1):
        current_text = ((text[i:i + n]).lower())
        res.append(current_text)

    return dict(Counter(res))


def read_data(n_terke):
    # Prosim, ne spreminjajte te funkcije. Vso potrebno obdelavo naredite
    # v funkciji terke.
    lds = {}
    for fn in listdir("jeziki"):
        if fn.lower().endswith(".txt"):
            with open(join("jeziki", fn), encoding="utf8") as f:
                text = f.read()
                nter = terke(text, n=n_terke)
                lds[fn] = nter
    return lds


def cosine_dist(d1, d2):
    """
    Vrne kosinusno razdaljo med slovarjema terk d1 in d2.
    """
    same_keys = d1.keys() & d2.keys()
    # ce gre za 2 enaka slovarja, vrnem 0
    if len(same_keys) == len(list(d1.keys())):
        return 0

    if len(same_keys) == 0:
        return 1

    numerator = sum(d1[key] * d2[key] for key in same_keys)
    norm_d1 = math.sqrt(sum(val ** 2 for val in d1.values()))
    norm_d2 = math.sqrt(sum(val ** 2 for val in d2.values()))
    denominator = norm_d1 * norm_d2

    return 1 - numerator / denominator


def prepare_data_matrix(data_dict):
    """
    Return data in a matrix (2D numpy array), where each row contains triplets
    for a language. Columns should be the 100 most common triplets
    according to the idf (NOT the complete tf-idf) measure.
    """
    # list of triplet and it's idf value
    triplet_idf = []
    # set of triplets
    set_triplets = set()

    # loop through keys in dict, get all triplets
    for key in data_dict.keys():
        for inner_key in data_dict.get(key):
            set_triplets.add(inner_key)

    # calculate IDF for each triplet
    for triplet in set_triplets:
        idf = calculate_idf(triplet, data_dict)
        triplet_idf.append([triplet, idf])

    # sort the list by idf value, ascending, 100 most common words
    triplet_idf.sort(key=lambda x: (x[1], x[0]), reverse=False)

    # print(triplet_idf[:100])

    # matrix: i-th row document, j-th column frequencies for j-th common triplet
    X = np.zeros(shape=(len(data_dict.keys()), 100))
    languages = list(data_dict.keys())

    # fill up the matrix X
    for i in range(len(data_dict.keys())):
        # get keys for current document
        keys_of_document = data_dict.get(languages[i])
        normalize = sum(keys_of_document.values())
        for j in range(min(100, len(set_triplets))):  # min, if we don't have enough triplets
            # get j-th most common triplet
            current_triplet = triplet_idf[j][0]
            # if document has current triplet
            if current_triplet in keys_of_document.keys():
                X[i][j] = keys_of_document.get(current_triplet) / normalize
                # print(X[i][j])

    return X, languages


def calculate_idf(triplet, data_dict):
    """
    Returns idf measure of current triplet.
    """
    frequency = 1
    for key in data_dict.keys():
        if triplet in data_dict.get(key).keys():
            frequency += 1

    return np.log(len(data_dict.keys()) / frequency)


def power_iteration(X):
    """
    Compute the eigenvector with the greatest eigenvalue
    of the covariance matrix of X (a numpy array).

    Return two values:
    - the eigenvector (1D numpy array) and
    - the corresponding eigenvalue (a float)
    """

    # center data
    X = X - np.mean(X, axis=0)
    # covariance
    X_cov = 1 / (X.shape[0] - 1) * np.dot(X.T, X)

    initial_vec = np.ones(X.shape[1])
    # print(initial_vec)
    diff = 1
    previous = initial_vec.copy()

    while diff > 0.00001:
        # matrix vector product
        matrix_vec_product = np.dot(X_cov, initial_vec)
        # calculate norm
        norm = np.linalg.norm(matrix_vec_product)
        # normalize
        initial_vec = matrix_vec_product / norm

        diff = np.linalg.norm(previous - initial_vec)
        # x = previous - initial_vec
        # diff = np.sqrt(x.dot(x))
        previous = initial_vec.copy()

    # get eigenvalue
    eigenvalue = np.dot(initial_vec.T, np.dot(X_cov, initial_vec))
    return initial_vec, eigenvalue


def power_iteration_two_components(X):
    """
    Compute first two eigenvectors and eigenvalues with the power iteration method.
    This function should use the power_iteration function internally.

    Return two values:
    - the two eigenvectors (2D numpy array, each eigenvector in a row) and
    - the corresponding eigenvalues (a 1D numpy array)
    """
    vec1, val1 = power_iteration(X)
    vec = vec1[..., np.newaxis]
    # project to eigenvector
    projection = X @ vec
    # get back matrix and subtract
    X_1 = X - projection @ vec.T
    # compute the next eigenvector, eigenvalue
    vec2, val2 = power_iteration(X_1)
    # print(vec1)

    A = np.zeros(shape=(2, len(vec1)))
    A[0] = vec1
    A[1] = vec2
    # print(A.T)

    return A, [val1, val2]


def project_to_eigenvectors(X, vecs):
    """
    Project matrix X onto the space defined by eigenvectors.
    The output array should have as many rows as X and as many columns as there
    are vectors.
    """
    X_1 = X.copy()
    X_1 -= np.mean(X_1, axis=0)

    return X_1 @ vecs.T


def total_variance(X):
    """
    Total variance of the data matrix X. You will need to use for
    to compute the explained variance ratio.
    """
    return np.var(X, axis=0, ddof=1).sum()


def explained_variance_ratio(X, eigenvectors, eigenvalues):
    """
    Compute explained variance ratio.
    """
    sum_eigenvalues = sum(eigenvalues)
    return sum_eigenvalues / total_variance(X)


def create_distance_matrix(X, keys):
    # initialize matrix
    matrix = np.zeros(shape=(len(keys), len(keys)))

    # calculate cosine distance between documents, only half the matrix
    for i in range(0, len(matrix)):
        doc_1 = X.get(keys[i])
        for j in range(i, len(matrix[0])):
            doc_2 = X.get(keys[j])
            dist = cosine_dist(doc_1, doc_2)
            matrix[i][j] = dist

    # fill the other half of the matrix, it's symmetrical
    matrix = matrix + matrix.T - np.diag(matrix.diagonal())
    return matrix


def plot_data(keys, X_transformed):
    # plt.figure()
    color_dict = {'ar': "red", 'ca': "green", 'de': "blue", 'en': "yellow", 'es': "pink", 'fa': "black", 'fr': "orange",
                  'hu': "purple", 'id': "gold", 'it': "brown", 'ja': "gray", 'nl': "cyan", 'pl': "magenta",
                  'pt': "olive", 'ru': "tan", 'sr': "plum", 'sv': "teal", 'uk': "khaki", 'zh': "salmon",
                  'vi': "skyblue"}
    for i, txt in enumerate(keys):
        txt = txt.split('.')[0]
        txt = txt.split('-')[1]
        color = color_dict.get(txt)
        plt.scatter(X_transformed[:, 0][i], X_transformed[:, 1][i], marker='*', c=color)
        plt.annotate(txt, (X_transformed[:, 0][i], X_transformed[:, 1][i]), fontsize=9)
    plt.show()
    return


def plot_PCA():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of PCA on languages data.
    """
    X = read_data(3)
    X, languages = prepare_data_matrix(X)
    vecs, val = power_iteration_two_components(X)
    # project

    value = explained_variance_ratio(X, vecs, val)
    X_transformed = project_to_eigenvectors(X, vecs)
    # print(X)

    res = "{:.2f}".format(value)
    plt.figure()
    plt.title('Explained variance: ' + res)

    plot_data(languages, X_transformed)
    return


def plot_MDS():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of MDS on languages data.

    Use sklearn.manifold.MDS and explicitly run it with a distance
    matrix obtained with cosine distance on full triplets.
    """
    # prebrani dokumenti ter njihove terke
    X = read_data(3)
    keys = list(X.keys())

    matrix = create_distance_matrix(X, keys)
    X_transformed = MDS(n_components=2, dissimilarity='precomputed').fit_transform(matrix)

    plt.figure()
    plt.title('MDS')
    plot_data(keys, X_transformed)
    return


def plot_tSNE():
    # prebrani dokumenti ter njihove terke
    X = read_data(3)
    keys = list(X.keys())

    matrix = create_distance_matrix(X, keys)
    X_transformed = TSNE(n_components=2, learning_rate='auto', init='random', metric='precomputed',
                         square_distances=True).fit_transform(matrix)

    plt.figure()
    plt.title('tSNE')
    plot_data(keys, X_transformed)
    return


if __name__ == "__main__":
    plot_MDS()
    plot_PCA()
    plot_tSNE()

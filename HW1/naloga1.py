import os
import random
import sys
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from numpy.linalg import norm
from torchvision import transforms


def read_data(path):
    """
    Preberi vse slike v podani poti, jih pretvori v "embedding" in vrni rezultat
    kot slovar, kjer so ključi imena slik, vrednosti pa pripadajoči vektorji.
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data = {}

    for image in os.listdir(path):

        input_image = Image.open(
            path + "\\" + image)  # .convert('RGB')  #, was used in testing, some images needed this

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        # index = image[5:-4]
        index = image
        data[index] = output[0]

    return data


def cosine_dist(d1, d2):
    """
    Vrni razdaljo med vektorjema d1 in d2, ki je smiselno
    pridobljena iz kosinusne podobnosti.
    """
    return 1 - np.dot(d1, d2) / (norm(d1) * norm(d2))

def k_medoids(data, medoids):
    """
    Za podane podatke (slovar vektorjev) in medoide vrni končne skupine
    kot seznam seznamov nizov (ključev v slovarju data).
    """
    clusters = {}

    # check if medoids given, if not generate them
    if not bool(medoids):
        # random.seed(1)
        medoids = random.sample(range(1, len(data.keys())), int(K))
        medoids = [list(data.keys())[x] for x in medoids]
    clusters = assign_points(data, medoids)

    convergence = True
    max_iter = 0
    initial_medoids = medoids.copy()

    while convergence and max_iter < np.inf:
        max_iter += 1

        # loop over current medoids
        for el in medoids:
            min_point = el
            # get elements which are assigned to current medoid
            current_list = clusters.get(el)

            # compute "cost" of current medoid
            cost_old = cost_function(el, current_list, data)

            # check for each element in list if it has a lower cos
            for point in current_list:
                cost_new = cost_function(point, current_list, data)

                if cost_new < cost_old:
                    min_point = point
                    cost_old = cost_new

            medoids[medoids.index(el)] = min_point

        clusters = assign_points(data, medoids)

        # check if continue or not
        if compare_medoids(initial_medoids, medoids):
            convergence = False
        else:
            initial_medoids = medoids.copy()
            convergence = True

    # return list of lists
    clusters = [clusters.get(key) for key in clusters]
    return clusters


# a function that checks if two sets of medoids are equal
def compare_medoids(initial_medoids, medoids):
    a = initial_medoids
    c = medoids
    a.sort()
    c.sort()
    return a == c


# assigns data points to given medoids
def assign_points(data, medoids):
    clusters = {}

    for image in data:
        min_dist = np.inf
        min_index = -1

        for el in medoids:
            distance = cosine_dist(data[image], data[el])
            if distance < min_dist:
                min_dist = distance
                min_index = el

        if min_index in clusters:
            clusters[min_index] = clusters.get(min_index) + [image]
        else:
            clusters[min_index] = [image]

    return clusters


# calculates the sum of distances from point to the points in the current_list
def cost_function(point, current_list, data):
    cost = 0
    for el in current_list:
        dist = cosine_dist(data[point], data[el])
        cost += dist
    return cost


def silhouette(el, clusters, data):
    """
    Za element el ob podanih podatke (slovar vektorjev) in skupinah
    (seznam seznamov nizov: ključev v slovarju data), vrni silhueto za element el.
    """
    # Find corresponding cluster, get cluster of given element
    cluster_of_el = []
    # get clusters that don't include the given element
    cluster_without_el = []

    for item in clusters:
        if el in item:
            cluster_of_el = item
        else:
            cluster_without_el.append(item)

    # Calculate a, compactness
    mean_a = cost_function(el, cluster_of_el, data)

    if len(cluster_of_el) == 1:
        mean_a = 0
        s = 0
        return s
    else:
        mean_a /= (len(cluster_of_el) - 1)

    # Calculate b, smallest mean of different cluster, most neighboring cluster
    min_mean_b = np.inf
    for item in cluster_without_el:
        # mean is the average value of distances to elements in current cluster
        mean = cost_function(el, item, data)
        mean /= (len(item))
        # saving the smaller mean
        if mean < min_mean_b:
            min_mean_b = mean

    # calculating the silhouette
    s = (min_mean_b - mean_a) / max(mean_a, min_mean_b)
    return s


def silhouette_average(data, clusters):
    """
    Za podane podatke (slovar vektorjev) in skupine (seznam seznamov nizov:
    ključev v slovarju data) vrni povprečno silhueto.
    """
    # Calculate silhouette for each element in the cluster
    sum = 0
    for group in clusters:
        for element in group:
            # gets the silhouette for given element and clusters
            sum += silhouette(element, clusters, data)

    return sum / len(data.keys())


if __name__ == "__main__":
    if len(sys.argv) == 3:
        K = sys.argv[1]
        path = sys.argv[2]
    else:
        K = 5
        path = "slike_4"

    # path = 'slike_4' Testing purposes
    data = read_data(path)
    best_avg = -np.inf
    best_clusters = 0
    best_K = 0

    idk = 0
    random.seed(30)
    # running k_medoids 100x times
    for i in range(0, 100):
        # K = 10 Testing purposes
        a = k_medoids(data, [])
        average = silhouette_average(data, a)

        # get the best groups based on silhouette score
        if best_avg < average:
            best_avg = average
            best_clusters = a
            best_K = K
            idk = i

    # Calculate silhouettes for the best clusters
    # store the image and silhouette in a list for representation
    images_with_scores = []
    for list in best_clusters:
        # print(f'Cluster: {list}')
        current = []
        for el in list:
            silh = silhouette(el, best_clusters, data)
            current.append([el, silh])
            # print(f'el: {el} with silhouette: {silh}')
        images_with_scores.append(current)

    # sort the list
    for list in images_with_scores:
        list.sort(key=lambda x: x[1], reverse=True)

    # display the lists
    cl = 1
    for list in images_with_scores:
        size = len(list)
        i = 0
        fig = plt.figure(figsize=(10, 5))

        for el in list:
            i += 1
            image = plt.imread(path + '/' + el[0])
            fig.add_subplot(3, 10, i)
            plt.imshow(image)
            plt.title(str(el[1])[:4])
            plt.axis('off')
            plt.savefig('cluster' + str(cl) + '.jpg')
        # plt.show()
        cl += 1
    # print(idk)
    # K = 2, 0.342
    # K = 3, 0.338
    # K = 4, 0.381
    # K = 5, 0.355
    # K = 6, 0.302
    # K = 7, 0.320
    # K = 8, 0.289
    # K = 9, 0.257
    # K = 10, 0.260

    # plotting graph of K
    # plt.plot([2, 3, 4, 5, 6, 7, 8, 9, 10], [0.342, 0.338, 0.381, 0.355, 0.302, 0.320, 0.289, 0.257, 0.260], "r")
    # plt.plot([2, 3, 4, 5, 6, 7, 8, 9, 10], [0.342, 0.338, 0.381, 0.355, 0.302, 0.320, 0.289, 0.257, 0.260], "bo")
    # plt.xlabel('K')
    # plt.ylabel('Silhouette average')
    # plt.show()

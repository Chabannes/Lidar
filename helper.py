import numpy as np
import matplotlib as mpl
import scipy.linalg
from laspy.file import File
import pandas as pd
import numba
#import plotly.express as px
#import plotly.graph_objects as go
import matplotlib.pyplot as plt
from numba import jit
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, prange
import itertools


def get_reduction(file, xmax=None, ymax=None, xmin=None, ymin=None):

    """
    :param file: string - nom du fichier LAS à traiter
    :param xmax: float - coordonnée xmax d'une zone réduite à étudier (optionnel)
    :param ymax: float - coordonnée ymax d'une zone réduite à étudier (optionnel)
    :param xmin: float - coordonnée xmin d'une zone réduite à étudier (optionnel)
    :param ymin: float - coordonnée ymin d'une zone réduite à étudier (optionnel)
    :return: Panda DataFrame, nuage de points contenu dans le fichier
    """

    X = file.x
    Y = file.y
    Z = file.z
    data = pd.DataFrame([X, Y, Z]).T
    data.columns = ['X', 'Y', 'Z']
    if xmax is not None:
        condition1 = data['X'] > xmin
        condition2 = data['X'] < xmax
        condition3 = data['Y'] > ymin
        condition4 = data['Y'] < ymax

        data = data[condition1 & condition2 & condition3 & condition4]
    return data


def get_reduction_with_angle(file, xmax=None, ymax=None, xmin=None, ymin=None):

    """
    :param file: string - nom du fichier LAS à traiter
    :param xmax: float - coordonnée xmax d'une zone réduite à étudier (optionnel)
    :param ymax: float - coordonnée ymax d'une zone réduite à étudier (optionnel)
    :param xmin: float - coordonnée xmin d'une zone réduite à étudier (optionnel)
    :param ymin: float - coordonnée ymin d'une zone réduite à étudier (optionnel)
    :return: Panda DataFrame, nuage de points contenu dans le fichier
    """
    X = file.x
    Y = file.y
    Z = file.z
    Angle = file.scan_angle
    data = pd.DataFrame([X, Y, Z, Angle]).T
    data.columns = ['X', 'Y', 'Z', 'Angle']

    if xmax is not None:
        condition1 = data['X'] > xmin
        condition2 = data['X'] < xmax
        condition3 = data['Y'] > ymin
        condition4 = data['Y'] < ymax

        data = data[condition1 & condition2 & condition3 & condition4]
    return data


@numba.jit
def projection_on_plane(data, coeff_plane, n):
    """
    :param data: numpy array - nuage de point à projeter sur un plan
    :param coeff_plane: list - coefficients définissant le plan
    :param n: int - nombre de points
    :return: list - list contenant les distances au plan de chaque point
    """
    projection = []
    for k in range(n):
        projection.append(abs(data[k, 0]*coeff_plane[0] + data[k, 1]*coeff_plane[1] - data[k, 2] + coeff_plane[2]) / \
           np.sqrt(coeff_plane[0]**2 + coeff_plane[1]**2 + 1))

    return projection



def local_distance_to_plane(x, y, cell_size, file, data=None):

    """
    :param x: float - coordonnée en x du centre da la cellule
    :param y: float - coordonnée en y du centre da la cellule
    :param size_cell: float - taille de cellule où sera calculé un plan fittant le nuage de point local. Surface size_cell x size_cell
    :param file: string - nom du fichier LAS à traiter
    :param data: Panda DataFrame - nuage de points d'une zone réduite de file (optionnel)
    :return: list - distances au plan de chaque point, float - moyenne de la liste projection, float - ecart type de la liste projection
            None si le nombre de point dans la cellule est insuffisant
    """

    try:  # si assez de données sur la zone choisie
        data = local_cloud(x, y, cell_size, file, data).values
        coeff = create_local_plane(data)
        n = data.shape[0]
        projection = projection_on_plane(data, coeff, n)

        return projection, np.mean(projection), np.std(projection)

    except:

        return None, None, None


def local_distance_to_plane_with_angle(x, y, cell_size, file, data=None):

    """
    :param x: float - coordonnée en x du centre da la cellule
    :param y: float - coordonnée en y du centre da la cellule
    :param size_cell: float - taille de cellule où sera calculé un plan fittant le nuage de point local. Surface size_cell x size_cell
    :param file: string - nom du fichier LAS à traiter
    :param data: Panda DataFrame - nuage de points d'une zone réduite de file (optionnel)
    :return: list - distances au plan de chaque point, float - moyenne de la liste projection, float - ecart type de la liste projection
            None si le nombre de point dans la cellule est insuffisant
    """

    try:  # si assez de données sur la zone choisie
        data, mean_angle = local_cloud_with_angle(x, y, cell_size, file, data)
        data = data.values
        coeff = create_local_plane(data)
        n = data.shape[0]
        projection = projection_on_plane(data, coeff, n)
        mean_angle = mean_angle*180/30000

        return projection, np.mean(projection), np.std(projection), mean_angle

    except:
        return None, None, None, None


def local_cloud(x, y, cell_size, file, data=None):

    """
    :param x: float - coordonnée en x du centre da la cellule
    :param y: float - coordonnée en y du centre da la cellule
    :param size_cell: float - taille de cellule où sera calculé un plan fittant le nuage de point local. Surface size_cell x size_cell
    :param file: string - nom du fichier LAS à traiter
    :param data: Panda DataFrame - nuage de points d'une zone réduite de file (optionnel)
    :return: Panda DataFrame - nuage de point contenu dans la cellule
            None si le nombre de points est insuffisant

    """
    try:  # si assez de données sur la zone choisie
        empty_data = data.empty

    except:

        empty_data = True

    if not empty_data:
        data_local = data
        condition1 = data_local['X'] > x - cell_size
        condition2 = data_local['X'] < x + cell_size
        condition3 = data_local['Y'] > y - cell_size
        condition4 = data_local['Y'] < y + cell_size
        new_data = data_local[condition1 & condition2 & condition3 & condition4]

    else:
        X = file.x
        Y = file.y
        Z = file.z
        local_data = pd.DataFrame([X, Y, Z]).T
        local_data.columns = ['X', 'Y', 'Z']
        condition1 = local_data['X'] > x - cell_size
        condition2 = local_data['X'] < x + cell_size
        condition3 = local_data['Y'] > y - cell_size
        condition4 = local_data['Y'] < y + cell_size
        new_data = local_data[condition1 & condition2 & condition3 & condition4]

    if new_data.shape[0] < 30:

        return None

    else:

        return new_data


def local_cloud_with_angle(x, y, cell_size, file, data=None):

    """
    :param x: float - coordonnée en x du centre da la cellule
    :param y: float - coordonnée en y du centre da la cellule
    :param size_cell: float - taille de cellule où sera calculé un plan fittant le nuage de point local. Surface size_cell x size_cell
    :param file: string - nom du fichier LAS à traiter
    :param data: Panda DataFrame - nuage de points d'une zone réduite de file (optionnel)
    :return: Panda DataFrame - nuage de point contenu dans la cellule avec les angles d'émission respectifs
            None si le nombre de points est insuffisant
    """

    try:  # si un dataframe data est entré par l'utilisateur
        empty_data = data.empty

    except:
        empty_data = True

    if not empty_data:
        local_data = data
        condition1 = local_data['X'] > x - cell_size
        condition2 = local_data['X'] < x + cell_size
        condition3 = local_data['Y'] > y - cell_size
        condition4 = local_data['Y'] < y + cell_size
        new_data = data[condition1 & condition2 & condition3 & condition4]

    else:
        X = file.x
        Y = file.y
        Z = file.z
        local_data = pd.DataFrame([X, Y, Z]).T
        local_data.columns = ['X', 'Y', 'Z']
        condition1 = local_data['X'] > x - cell_size
        condition2 = local_data['X'] < x + cell_size
        condition3 = local_data['Y'] > y - cell_size
        condition4 = local_data['Y'] < y + cell_size
        new_data = local_data[condition1 & condition2 & condition3 & condition4]

    if new_data.shape[0] < 30:

        return None, None

    else:

        return new_data, new_data['Angle'].mean()


def create_local_plane(data):

    """
    :param data: numpy array - nuage de point devant être fitté par un plan
    :return: list - coefficients definissant le plan
    """

    # best-fit linear plane
    A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    Coeff, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

    return Coeff


def get_vecteur_normal(x, y, file, cell_size, data):

    """
    :param x: float - coordonnée en x du centre da la cellule
    :param y: float - coordonnée en y du centre da la cellule
    :param size_cell: float - taille de cellule où sera calculé un plan fittant le nuage de point local. Surface size_cell x size_cell
    :param file: string - nom du fichier LAS à traiter
    :param data: Panda DataFrame - nuage de points d'une zone réduite de file (optionnel)
    :return: numpy array - coordonnées définissant le vecteur normal au plan fittant le nuage de point entré
        None si le nombre de point dans le nuage de point est trop faible
    """

    try:  # si assez de données sur la zone choisie
        data_cell = local_cloud(x, y, cell_size, file, data).values
        # regular grid covering the domain of the data

        # best-fit du nuage de point par un plan
        A = np.c_[data_cell[:, 0], data_cell[:, 1], np.ones(data_cell.shape[0])]
        Coeff, _, _, _ = scipy.linalg.lstsq(A, data_cell[:, 2])  # coefficients
        n = np.array([Coeff[0], Coeff[1], -1])

        return n

    except:

        return None


def plot_distance_to_local_plane(x, y, cell_size, file):

    """
    :param x: float - coordonnée en x du centre da la cellule
    :param y: float - coordonnée en y du centre da la cellule
    :param size_cell: float - taille de cellule où sera calculé un plan fittant le nuage de point local. Surface size_cell x size_cell
    :param file: string - nom du fichier LAS à traiter
    :return: float - moyenne et ecarts types des distances des points au plan, plot l'histogramme
            None si nombre de point dans le nuage de point est insuffisant
    """

    if local_distance_to_plane(x, y, cell_size, file)[0] == None:

        print('Echec construction histogramme : pas assez de données')

        return None, None
    else :
        projection, mean, std = local_distance_to_plane(x, y, cell_size, file)
        plt.figure()
        plt.title('Histogram of distance to local plane \nfile name : file %s' % str(file))
        plt.hist(projection, bins=200, color='red')
        plt.xlabel('Distance to local plane [m]')
        plt.ylabel('Repartition')
        plt.title("File : %s\nMean distance : %s m \nStandard deviation :%s m"\
                  % (np.round(mean, 4), np.round(std, 4)), fontsize=10)
        plt.legend()

        return mean, std


@numba.jit
def calcul_pdf(X, Y, n_sample, cell_size, file, data):

    """
    :param X: numpy array - coordonnées en x du nuage de point
    :param Y numpy array - coordonnées en y du nuage de point
    :param n_sample: int - nombre de cellule à tirer aléatoirement
    :param size_cell: float - taille de cellule où sera calculé un plan fittant le nuage de point local. Surface size_cell x size_cell
    :param file: string - nom du fichier LAS à traiter
    :return: list - liste de longueur n contenant les distances moyennes des n nuages de points à leurs plans respectifs
    """

    list_mean = []
    for compteur in range(n_sample):
        x, y = np.random.uniform(np.min(X) + cell_size, np.max(X) - cell_size), np.random.uniform(np.min(Y) + cell_size, np.max(Y) - cell_size)
        projection, mean, std = local_distance_to_plane(x, y, cell_size, file, data)
        print("Point : ", compteur, "/", n_sample)
        if mean is not None:
            list_mean.append(mean)

    return list_mean


@numba.jit
def calcul_pdf_allez_retour(X, Y, n_sample, cell_size, file1, file2, data1=None, data2=None):

    """
    :param X: numpy array - coordonnées en x du nuage de point
    :param Y numpy array - coordonnées en y du nuage de point
    :param n_sample: int - nombre de cellule à tirer aléatoirement
    :param size_cell: float - taille de cellule où sera calculé un plan fittant le nuage de point local. Surface size_cell x size_cell
    :param file1: string - nom du fichier LAS à traiter correspondant au trajet aller
    :param file2: string - nom du fichier LAS à traiter correspondant au trajet retour
    :param data1: Panda DataFrame - nuage de points d'une zone réduite de file1 (optionnel)
    :param data2: Panda DataFrame - nuage de points d'une zone réduite de file2 (optionnel)
    :return: list - deux listes contenant les distances moyennes aux plans, pour l'aller et le retour
    """


    list_mean1 = []
    list_mean2 = []
    for compteur in range(n_sample):
        x, y = np.random.uniform(np.min(X) + cell_size, np.max(X) - cell_size), np.random.uniform(np.min(Y) + cell_size, np.max(Y) - cell_size)

        projection1, mean1, std1 = local_distance_to_plane(x, y, cell_size, file1, data1)
        projection2, mean2, std2 = local_distance_to_plane(x, y, cell_size, file2, data2)

        print("Point : ", compteur, "/", n_sample)
        if mean1 is not None:

            list_mean1.append(mean1)

        if mean2 is not None:
            list_mean2.append(mean2)

    return list_mean1, list_mean2


def plot_3D(file1, file2, reduction, xmax=None, ymax=None, xmin=None, ymin=None):

    """
    :param file1: string - nom du fichier LAS à traiter correspondant au trajet aller
    :param file2: string - nom du fichier LAS à traiter correspondant au trajet retour
    :param reduction: int - coefficient de reduction du nombre de points. Pour reduction = 2, on prend un point sur deux
    :param xmax: float - coordonnée xmax d'une zone réduite à étudier (optionnel)
    :param ymax: float - coordonnée ymax d'une zone réduite à étudier (optionnel)
    :param xmin: float - coordonnée xmin d'une zone réduite à étudier (optionnel)
    :param ymin: float - coordonnée ymin d'une zone réduite à étudier (optionnel)
    :return: None, plot la surface 3D correspondant au trajet aller et retour
    """

    X1 = file1.x[::reduction]
    Y1 = file1.y[::reduction]
    Z1 = file1.z[::reduction]
    X2 = file2.x[::reduction]
    Y2 = file2.y[::reduction]
    Z2 = file2.z[::reduction]

    data1 = pd.DataFrame([X1, Y1, Z1]).T
    data1.columns = ['X', 'Y', 'Z']

    data2 = pd.DataFrame([X2, Y2, Z2]).T
    data2.columns = ['X', 'Y', 'Z']

    if xmax is not None:
        condition1 = data1['X'] > xmin
        condition2 = data1['X'] < xmax
        condition3 = data1['Y'] > ymin
        condition4 = data1['Y'] < ymax

        data1 = data1[condition1 & condition2 & condition3 & condition4]

        condition1 = data2['X'] > xmin
        condition2 = data2['X'] < xmax
        condition3 = data2['Y'] > ymin
        condition4 = data2['Y'] < ymax

        data2 = data2[condition1 & condition2 & condition3 & condition4]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(data1['X'], data1['Y'], data1['Z'], alpha=0.8, color='b', s=0.05, label='Trajet aller')
    ax.scatter(data2['X'], data2['Y'], data2['Z'], alpha=0.8, color='r', s=0.05, label='Trajet retour')
    ax.legend()
    ax.set_xlabel('X[m]')
    ax.set_ylabel('Y[m]')
    ax.set_zlabel('Z[m]')


def pca_results(good_data, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    # Dimension indexing
    dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = list(good_data.keys()))
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize = (14,8))

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar')
    ax.set_ylabel("Poids des features")
    ax.set_xticklabels(dimensions, rotation=0)


    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis=1)


def biplot(good_data, reduced_data, pca):
    '''
    Produce a biplot that shows a scatterplot of the reduced
    data and the projections of the original features.

    good_data: original data, before transformation.
               Needs to be a pandas dataframe with valid column names
    reduced_data: the reduced data (the first two dimensions are plotted)
    pca: pca object that contains the components_ attribute

    return: a matplotlib AxesSubplot object (for any additional customization)

    This procedure is inspired by the script:
    https://github.com/teddyroland/python-biplot
    '''


    fig, ax = plt.subplots(figsize=(25, 15))
    # scatterplot of the reduced data
    ax.scatter(x=reduced_data.loc[:, 'Dimension 1'], y=reduced_data.loc[:, 'Dimension 2'],
               facecolors='b', edgecolors='b', s=0.3, alpha=0.5)
    from sklearn import mixture
    feature_vectors = pca.components_.T

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 0.3, 0.7

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size * v[0], arrow_size * v[1],
                 head_width=0.01, head_length=0.02, linewidth=2, color='red')
        ax.text(v[0] *text_pos, v[1] *text_pos, good_data.columns[i], color='black',
                ha='center', va='center', fontsize=11)

    ax.set_xlabel("Composante 1", fontsize=14)
    ax.set_ylabel("Composante 2", fontsize=14)
    ax.set_title("Projection des données sur les deux premières composantes de PCA", fontsize=12)
    plt.show()
    return ax




def plot_resultsGMM(X, Y_, means, covariances, index, title):

    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                                  'darkorange'])
    title = 'Clustering des erreurs Lidar'
    labels = ['Cluster 1', 'Cluster 2']
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = np.linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])

        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .1, color=color, label=labels[i])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.legend()
    plt.xlabel('Composante 1')
    plt.ylabel('Composante 2')
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

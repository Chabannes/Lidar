import numpy as np
import scipy.linalg
import scipy.stats as stats
import laspy
from laspy.file import File
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import random
import numba
from scipy.stats import norm
from scipy.optimize import curve_fit
from helper import *
from numba import njit, prange
import numpy.polynomial.polynomial as poly
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.decomposition import PCA
from statistics import mode



def main():
    file_water1 = File("SAMPLE CLOUDS/WATER_L180_00utm.las", mode="r")
    file_water2 = File("SAMPLE CLOUDS/WATER_L180_02utm.las", mode="r")

    file_sand1 = File("SAMPLE CLOUDS/SAND_L180_00utm.las", mode="r")
    file_sand2 = File("SAMPLE CLOUDS/SAND_L180_02utm.las", mode="r")

    file_tree1 = File("SAMPLE CLOUDS/TREE_L180_00utm.las", mode="r")
    file_tree2 = File("SAMPLE CLOUDS/TREE_L180_02utm.las", mode="r")

    file_toit1 = File("SAMPLE CLOUDS/toit1.las", mode="r")
    file_toit2 = File("SAMPLE CLOUDS/toit2.las", mode="r")

    file_berge_aller = File("DONNEES/BERGES/BERGE_180_00_UTM.las", mode="r")
    file_berge_retour = File("DONNEES/BERGES/BERGE_180_01_UTM.las", mode="r")

    file_berge_retour_plan = File("DONNEES/BERGES/BERGE_PLAN_180_01_UTM.las", mode="r")
    #file_berge_aller_plan = File("DONNEES/BERGES/BERGE_PLAN_180_00_UTM.las", mode="r")


    file_toit1_roulis = File("SAMPLE CLOUDS/roulis1.las", mode="r")
    file_toit2_roulis = File("SAMPLE CLOUDS/roulis2.las", mode="r")

    #file_ref = File("SAMPLE CLOUDS/ref_toit_utm.las", mode="r")
    #file_test_by_ref = File("SAMPLE CLOUDS/toit_lid_utm.las", mode="r")

    reduction = 10
    cell_size = 0.8
    n_samples = 70000

    # coordonnées réduction facade toit : xmax=498609.4, ymax=5337832.2, xmin=498606.7, ymin=5337828.
    # coordonnées réduction facade toit total : xmax=4986012.4, ymax=5337855, xmin=498590.7, ymin=5337720.3


    #plot_3D(file_berge_aller_plan, file_berge_retour_plan, reduction) #, xmax=4986012.4, ymax=5337855, xmin=498590.7, ymin=5337720.3)
    #erreur_aleatoire(file_toit1, cell_size, n_samples, xmax=498610.4, ymax=5337831.7, xmin=498606.7, ymin=5337829.3)
    #erreur_aleatoire_aller_retour(file_toit1, file_toit2, cell_size, n_samples, xmax=498614.4, ymax=5337835.7, xmin=498606.7, ymin=5337829.3)
    evaluation_boresight(file_berge_aller, file_berge_retour_plan, cell_size, n_samples) #, xmax=4986012.4, ymax=5337855, xmin=498590.7, ymin=5337720.3)
    #evaluer_erreur_aleatoire_angle(file_toit1, cell_size, n_samples) #  , xmax=498609.4, ymax=5337832.2, xmin=498606.7, ymin=5337828.3)
    #erreur_absolue(file_test_by_ref, file_ref, n_samples, cell_size, Graphique=True)#, xmax=498609.4, ymax=5337832.2, xmin=498606.7, ymin=5337828.3)
    #evaluer_erreur_absolue_angle(file_test_by_ref, file_ref, n_samples, cell_size, xmax=None, ymax=None, xmin=None, ymin=None)
    #clustering_erreur(file_test_by_ref, file_ref, cell_size, n_samples)

    plt.show()


def erreur_aleatoire(file, cell_size, n_sample, xmax=None, ymax=None, xmin=None, ymin=None):

    """
    :param file: string - nom du fichier LAS à traiter
    :param size_cell: float - taille de cellule où sera calculé un plan fittant le nuage de point local. Surface size_cell x size_cell
    :param n_sample: int - nombre de cellules choisies aléatoirement
    :param xmax: float - coordonnée xmax d'une zone réduite à étudier (optionnel)
    :param ymax: float - coordonnée ymax d'une zone réduite à étudier (optionnel)
    :param xmin: float - coordonnée xmin d'une zone réduite à étudier (optionnel)
    :param ymin: float - coordonnée ymin d'une zone réduite à étudier (optionnel)
    :return: None, plot l'histogramme des erreurs aléatoires
    """

    data = get_reduction(file, xmax, ymax, xmin, ymin)
    list_mean = calcul_pdf(data['X'], data['Y'], n_sample, cell_size, file, data)

    list_mean = np.asarray(list_mean)
    mu, std = norm.fit(list_mean)

    plt.figure()
    ax = sns.distplot(list_mean, kde=False, bins=int(len(list_mean)/10), label="Moyenne = %.3fm,  Ecart-type = %.4fm" % (mu, std))
    ax.legend()
    ax.set_title("Distribution de l'erreur aléatoire")
    ax.set_xlabel('Erreur aléatoire [m]')


def erreur_aleatoire_aller_retour(file1, file2,  cell_size, n_sample, xmax=None, ymax=None, xmin=None, ymin=None):

    """
    :param file1: string - nom du fichier LAS à traiter correspondant au trajet aller
    :param file2: string - nom du fichier LAS à traiter correspondant au trajet retour
    :param size_cell: float - taille de cellule où sera calculé un plan fittant le nuage de point local. Surface size_cell x size_cell
    :param n_sample: int - nombre de cellules choisies aléatoirement
    :param xmax: float - coordonnée xmax d'une zone réduite à étudier (optionnel)
    :param ymax: float - coordonnée ymax d'une zone réduite à étudier (optionnel)
    :param xmin: float - coordonnée xmin d'une zone réduite à étudier (optionnel)
    :param ymin: float - coordonnée ymin d'une zone réduite à étudier (optionnel)
    :return: None, plot les histogrammes des erreurs aléatoires aller et retour
    """

    data1 = get_reduction(file1, xmax, ymax, xmin, ymin)
    data2 = get_reduction(file2, xmax, ymax, xmin, ymin)

    list_mean1, list_mean2 = calcul_pdf_allez_retour(data1['X'], data1['Y'], n_sample, cell_size, file1, file2, data1, data2)

    list_mean1, list_mean2 = np.asarray(list_mean1), np.asarray(list_mean2)
    mu1, std1 = norm.fit(list_mean1)
    mu2, std2 = norm.fit(list_mean2)
    plt.figure()

    plt.hist(list_mean1, bins=int(len(list_mean1)/5), label="Moyenne aller = %.3fm,  écart-type aller = %.4fm" % (mu1, std1), alpha=0.5)
    plt.hist(list_mean2, bins=int(len(list_mean2)/5), label="Moyenne retour = %.3fm,  écart-type retour = %.4fm" % (mu2, std2), alpha=0.5)

    plt.legend(loc='upper right')
    plt.title("Densité de probabilité de l'erreur aléatoire")
    plt.xlabel('Erreur aléatoire [m]')



def evaluation_boresight(file1, file2,  cell_size, n_sample, xmax=None, ymax=None, xmin=None, ymin=None):

    """
    :param file1: string - nom du fichier LAS à traiter correspondant au trajet aller
    :param file2: string - nom du fichier LAS à traiter correspondant au trajet retour
    :param size_cell: float - taille de cellule où sera calculé un plan fittant le nuage de point local. Surface size_cell x size_cell
    :param n_sample: int - nombre de cellules choisies aléatoirement
    :param xmax: float - coordonnée xmax d'une zone réduite à étudier (optionnel)
    :param ymax: float - coordonnée ymax d'une zone réduite à étudier (optionnel)
    :param xmin: float - coordonnée xmin d'une zone réduite à étudier (optionnel)
    :param ymin: float - coordonnée ymin d'une zone réduite à étudier (optionnel)
    :return: None, plot l'histogramme des angles de boresight
    """
    data1 = get_reduction(file1, xmax, ymax, xmin, ymin)
    data2 = get_reduction(file2, xmax, ymax, xmin, ymin)


    liste_roulis = []
    liste_tangage = []

    for compteur in range(0, n_sample):
        X = data1['X']
        Y = data1['Y']
        ex = np.array([1, 0, 0])
        ey = np.array([0, 1, 0])
        ez = np.array([0, 0, 1])
        x, y = np.random.uniform(np.min(X), np.max(X)), np.random.uniform(np.min(Y), np.max(Y))
        n1 = get_vecteur_normal(x, y, file1, cell_size, data1)
        n2 = get_vecteur_normal(x, y, file1, cell_size, data2)
        print(compteur, "/", n_sample)

        try:  # si assez de données sur la zone choisie

            # selon x
            n1x = np.dot(n1, ey)*ey + np.dot(n1, ez)*ez
            n2x = np.dot(n2, ey)*ey + np.dot(n2, ez)*ez

            if np.dot((np.cross(n1x, n2x)), ex) > 0:
                angle_roulis = (np.arccos(np.dot(n1x, n2x) / (np.linalg.norm(n1x)*np.linalg.norm(n2x)))) #*180/np.pi

            if np.dot((np.cross(n1x, n2x)), ex) < 0:
                angle_roulis = (-np.arccos(np.dot(n1x, n2x) / (np.linalg.norm(n1x)*np.linalg.norm(n2x)))) #*180/np.pi

            liste_roulis.append(angle_roulis)

            # selon y
            n1y = np.dot(n1, ex)*ex + np.dot(n1, ez)*ez
            n2y = np.dot(n2, ex)*ex + np.dot(n2, ez)*ez

            if np.dot((np.cross(n1y, n2y)), ey) > 0:
                angle_tangage = (np.arccos(np.dot(n1y, n2y) / (np.linalg.norm(n1y) * np.linalg.norm(n2y)))) #*180/np.pi

            if np.dot((np.cross(n1y, n2y)), ey) < 0:
                angle_tangage = (-np.arccos(np.dot(n1y, n2y) / (np.linalg.norm(n1y) * np.linalg.norm(n2y)))) #*180/np.pi

            liste_tangage.append(angle_tangage)

        except:
            pass

    fig, ax = plt.subplots()
    mu_roulis, std_roulis = np.mean(liste_roulis), np.std(liste_roulis)
    #sns.distplot(liste_roulis, bins=int(len(liste_roulis)/3), kde=False, label="mu = %.3f°\nstd =  %.3f°" % (mu_roulis, std_roulis), color='b')
    n, b, patches = plt.hist(liste_roulis, bins=int(len(liste_roulis)/2), color='blue', histtype='stepfilled')
    bin_max = np.where(n == n.max())
    mode1 = b[bin_max][0]
    s = "mode roulis = " + str(round(mode1,4)) + " °"
    plt.plot(0,0, c='b', label=s)
    plt.xlabel('Angle de roulis [°]')
    plt.title('Distribution des mesures de désalignement en roulis')
    plt.legend()


    fig, ax = plt.subplots()
    mu_tangage, std_tangage = np.mean(liste_tangage), np.std(liste_tangage)
    n, b, patches = plt.hist(liste_tangage, bins=int(len(liste_tangage)/2), color='blue', histtype='stepfilled')
    bin_max = np.where(n == n.max())
    mode2 = b[bin_max][0]
    s = "mode tangage = " + str(round(mode2,4)) + " °"
    plt.plot(0,0, c='b', label=s)
    plt.xlabel('Angle de tangage [°]')
    plt.title('Distribution des mesures de désalignement en tangage')
    plt.legend()



def evaluer_erreur_aleatoire_angle(file, cell_size, n_sample, xmax=None, ymax=None, xmin=None, ymin=None):

    """
    :param file: string - nom du fichier LAS à traiter
    :param size_cell: float - taille de cellule où sera calculé un plan fittant le nuage de point local. Surface size_cell x size_cell
    :param n_sample: int - nombre de cellules choisies aléatoirement
    :param xmax: float - coordonnée xmax d'une zone réduite à étudier (optionnel)
    :param ymax: float - coordonnée ymax d'une zone réduite à étudier (optionnel)
    :param xmin: float - coordonnée xmin d'une zone réduite à étudier (optionnel)
    :param ymin: float - coordonnée ymin d'une zone réduite à étudier (optionnel)
    :return: None, plot les erreurs aléatoires en fonction de l'angle absolu d'émission
    """

    data = get_reduction_with_angle(file, xmax, ymax, xmin, ymin)

    X = data['X']
    Y = data['Y']

    list_mean = []
    list_angle = []

    for compteur in range(0, n_sample):
        x, y = np.random.uniform(np.min(X) + cell_size, np.max(X) - cell_size), np.random.uniform(np.min(Y) + cell_size, np.max(Y) - cell_size)
        projection, mean_error, std, mean_angle = local_distance_to_plane_with_angle(x, y, cell_size, file, data)
        print("Point : ", compteur, "/", n_sample)

        if mean_error != None and mean_angle != None:
            list_mean.append(mean_error*100)  # erreur aleatoire en cm
            list_angle.append(mean_angle)

    list_angle_sorted, list_mean_sorted = (list(t) for t in zip(*sorted(zip(list_angle, list_mean))))
    list_angle_sorted_flag = []
    list_mean_sorted_flag = []

    for k in range(len(list_angle_sorted)):
        if list_mean_sorted[k] < 8:
            list_mean_sorted_flag.append(list_mean_sorted[k])
            list_angle_sorted_flag.append(list_angle_sorted[k])

    list_angle_sorted = list_angle_sorted_flag
    list_mean_sorted = list_mean_sorted_flag

    def func(x, a, b, c):
        return a * x**2 + b*x + c

    popt, pcov = curve_fit(func, np.array(list_angle_sorted), np.array(list_mean_sorted))

    plt.figure()
    plt.title("Evolution de l'erreur aléatoire en fonction de l'angle d'émission")
    plt.xlabel("Angle d'émission absolu [°]")
    plt.ylabel("Erreur aléatoire [cm]")
    plt.scatter(list_angle, list_mean, s=0.5)

    plt.figure()
    plt.title("Evolution de l'erreur aléatoire en fonction de l'angle d'émission")
    plt.xlabel("Angle d'émission absolu [°]")
    plt.ylabel("Erreur aléatoire [cm]")
    plt.plot(list_angle_sorted, func(np.array(list_angle_sorted), *popt), 'r-', label = 'fit: a=%5.4f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.legend()
    plt.scatter(list_angle_sorted, list_mean_sorted, s=0.5)


def erreur_absolue(file1, file_ref, n_sample, cell_size, Graphique=None, xmax=None, ymax=None, xmin=None, ymin=None):

    """
    :param file1: string - nom du fichier LAS à traiter (levé Lidar)
    :param file_ref: string - nom du fichier LAS correspondant au levé laser statique (référence)
    :param n_sample: int - nombre de cellules choisies aléatoirement
    :param size_cell: float - taille de cellule où sera calculé un plan fittant le nuage de point local. Surface size_cell x size_cell
    :param Graphique: bool - si vrai, un graphique est affiché montrant les deux plans fittants les points Lidar et laser statique
    :param xmax: float - coordonnée xmax d'une zone réduite à étudier (optionnel)
    :param ymax: float - coordonnée ymax d'une zone réduite à étudier (optionnel)
    :param xmin: float - coordonnée xmin d'une zone réduite à étudier (optionnel)
    :param ymin: float - coordonnée ymin d'une zone réduite à étudier (optionnel)
    :return:
    """

    data = get_reduction(file1, xmax, ymax, xmin, ymin)
    data_ref = get_reduction(file_ref, xmax, ymax, xmin, ymin)

    data_representation = False

    print(data.shape)
    print(data_ref.shape)

    rapport_reduction = int(np.shape(data_ref)[0] / np.shape(data)[0])

    if rapport_reduction != 0:
        data_ref = data_ref[::rapport_reduction]

    X = data['X']
    Y = data['Y']

    ### Erreur Absolue ###

    distance_plan = []
    for compteur in range(n_sample):
        print("Point : ", compteur, "/", n_sample)
        try:  # si assez de données sur la zone choisie
            x, y = np.random.uniform(np.min(X) + cell_size, np.max(X) - cell_size), np.random.uniform(np.min(Y) + cell_size, np.max(Y) - cell_size)

            local_data_ref = local_cloud(x, y, cell_size, file_ref, data=data_ref)
            local_data = local_cloud(x, y, cell_size, file1, data=data)

            xref, yref, zref = local_data_ref['X'].mean(),  local_data_ref['Y'].mean(), local_data_ref['Z'].mean()

            local_data_values = local_data.values
            A_lidar = np.c_[local_data_values[:, 0], local_data_values[:, 1], np.ones(local_data_values.shape[0])]
            Coeff, _, _, _ = scipy.linalg.lstsq(A_lidar, local_data_values[:, 2])
            z_lidar = Coeff[0] * xref + Coeff[1] * yref + Coeff[2]

            distance = zref - z_lidar#
            distance_plan.append(distance)

            if not data_representation:
                Local_data = local_data  # data utilisée pour la représentation graphique
                Local_data_ref = local_data_ref  # data utilisée pour la représentation graphique
                data_representation = True

        except:
            pass

    mu = np.mean(distance_plan)
    std = np.std(distance_plan)
    gauss = stats.norm.pdf(np.sort(distance_plan), mu, std)

    fig, ax = plt.subplots()
    plt.hist(distance_plan, bins=80, color='dodgerblue', normed=1)
    plt.plot(np.sort(distance_plan), gauss, color='k', label='Moyenne = %.3fm\nEcart-type = %.4fm' %(mu, std))
    ax.legend()
    ax.set_title("Erreur absolue - utilisation laser statique")
    ax.set_xlabel('Biais vertical [m]')

    ### ###


    ### Comparaison Erreur Relative ###


    list_mean1, list_mean2 = calcul_pdf_allez_retour(data['X'], data['Y'], n_sample, cell_size, file1, file_ref, data, data_ref)

    list_mean1, list_mean2 = np.asarray(list_mean1), np.asarray(list_mean2)
    mu1, std1 = norm.fit(list_mean1)
    mu2, std2 = norm.fit(list_mean2)

    fig, ax = plt.subplots()
    ax = sns.distplot(list_mean2, kde=False, bins=int(n_sample/6), label="Moyenne Laser Statique = %.3fm,  écart-type = %.4fm" % (mu2, std2))
    ax = sns.distplot(list_mean1, kde=False, bins=int(n_sample/6), label="Moyenne Lidar = %.3fm,  écart-type = %.4fm" % (mu1, std1))

    ax.legend(loc='lower right')
    ax.set_title("Distribution de l'erreur aléatoire pour Lidar et Laser Statique")
    ax.set_xlabel('Erreur aléatoire [m]')

    ### ###


    if Graphique:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(Local_data_ref['X'], Local_data_ref['Y'], Local_data_ref['Z'], alpha=1, color='b', s=0.2)
        ax.scatter(Local_data['X'], Local_data['Y'], Local_data['Z'], alpha=1, color='r', s=0.2)

        A_ref = np.c_[Local_data_ref.values[:, 0], Local_data_ref.values[:, 1], np.ones(Local_data_ref.shape[0])]
        Coeff_ref, _, _, _ = scipy.linalg.lstsq(A_ref, Local_data_ref.values[:, 2])
        X_ref = np.array([Local_data_ref['X'].min(), Local_data_ref['X'].max()])
        Y_ref = np.array([Local_data_ref['Y'].min(), Local_data_ref['Y'].max()])
        X_ref, Y_ref = np.meshgrid(X_ref, Y_ref)
        Z_ref = Coeff_ref[0] * X_ref + Coeff_ref[1] * Y_ref + Coeff_ref[2]

        A = np.c_[Local_data.values[:, 0], Local_data.values[:, 1], np.ones(Local_data.shape[0])]
        Coeff, _, _, _ = scipy.linalg.lstsq(A, Local_data.values[:, 2])
        X = np.array([Local_data['X'].min(), Local_data['X'].max()])
        Y = np.array([Local_data['Y'].min(), Local_data['Y'].max()])
        X, Y = np.meshgrid(X, Y)
        Z = Coeff[0] * X + Coeff[1] * Y + Coeff[2]

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5)
        ax.plot_surface(X_ref, Y_ref, Z_ref, rstride=1, cstride=1, alpha=0.5)


        col1_patch = mpatches.Patch(color ='blue', label='Référence laser statique')
        col2_patch = mpatches.Patch(color='brown', label='Nuage de points Lidar')

        plt.legend(handles=[col1_patch, col2_patch], loc='lower left')

        ax.set_title("Erreur absolue sur une cellule")
        ax.set_xlabel('X[m]')
        ax.set_ylabel('Y[m]')
        ax.set_zlabel('Z[m]')


def evaluer_erreur_absolue_angle(file1, file_ref, n_sample, cell_size, xmax=None, ymax=None, xmin=None, ymin=None):


    """
    :param file1: string - nom du fichier LAS à traiter (levé Lidar)
    :param file_ref: string - nom du fichier LAS correspondant au levé laser statique (référence)
    :param n_sample: int - nombre de cellules choisies aléatoirement
    :param size_cell: float - taille de cellule où sera calculé un plan fittant le nuage de point local. Surface size_cell x size_cell
    :param Graphique: bool - si vrai, un graphique est affiché montrant les deux plans fittants les points Lidar et laser statique
    :param xmax: float - coordonnée xmax d'une zone réduite à étudier (optionnel)
    :param ymax: float - coordonnée ymax d'une zone réduite à étudier (optionnel)
    :param xmin: float - coordonnée xmin d'une zone réduite à étudier (optionnel)
    :param ymin: float - coordonnée ymin d'une zone réduite à étudier (optionnel)
    :return:
    """

    data = get_reduction_with_angle(file1, xmax, ymax, xmin, ymin)
    data_ref = get_reduction(file_ref, xmax, ymax, xmin, ymin)
    rapport_reduction = int(np.shape(data_ref)[0] / np.shape(data)[0])

    if rapport_reduction != 0:
        data_ref = data_ref[::rapport_reduction]

    X = data['X']
    Y = data['Y']

    ### Erreur Absolue ###

    erreur_absolue = []
    liste_angle = []
    for compteur in range(n_sample):
        print("Point : ", compteur, "/", n_sample)
        try:  # si assez de données sur la zone choisie
            x, y = np.random.uniform(np.min(X) + cell_size, np.max(X) - cell_size), np.random.uniform(np.min(Y) + cell_size, np.max(Y) - cell_size)
            local_data_ref = local_cloud(x, y, cell_size, file_ref, data_ref)
            local_data, angle = local_cloud_with_angle(x, y, cell_size, file1, data)

            xref, yref, zref = local_data_ref['X'].mean(),  local_data_ref['Y'].mean(), local_data_ref['Z'].mean()

            local_data_values = local_data.values

            A_lidar = np.c_[local_data_values[:, 0], local_data_values[:, 1], np.ones(local_data_values.shape[0])]
            Coeff, _, _, _ = scipy.linalg.lstsq(A_lidar, local_data_values[:, 2])
            z_lidar = Coeff[0] * xref + Coeff[1] * yref + Coeff[2]
            distance = zref - z_lidar #

            erreur_absolue.append(distance*100)  # erreur absolue en cm
            liste_angle.append(angle*180/30000)  # convertir en degrés

        except:
            pass

    liste_angle_sorted, erreur_absolue_sorted = (list(t) for t in zip(*sorted(zip(liste_angle, erreur_absolue))))

    def func(x, a, b, c):
        return a * x**2 + b*x + c

    popt, pcov = curve_fit(func, np.array(liste_angle_sorted), np.array(erreur_absolue_sorted))

    plt.title("Evolution de l'erreur absolue en fonction de l'angle d'émission absolu")
    plt.scatter(liste_angle_sorted, erreur_absolue_sorted, s=0.1)
    plt.plot(liste_angle_sorted, func(np.array(liste_angle_sorted), *popt), 'r-', label = 'fit: a=%5.4f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.xlabel("Angle d'émission absolu [°]")
    plt.ylabel('Erreur Absolue [cm]')
    plt.legend()


def clustering_erreur(file, file_ref, cell_size, n_sample, xmax=None, ymax=None, xmin=None, ymin=None):

    columns = ['Angle','Erreur Absolue', 'Erreur Aleatoire']

    data = get_reduction_with_angle(file, xmax, ymax, xmin, ymin)
    data_ref = get_reduction(file_ref, xmax, ymax, xmin, ymin)

    rapport_reduction = int(np.shape(data_ref)[0] / np.shape(data)[0])

    if rapport_reduction != 0:
        data_ref = data_ref[::rapport_reduction]

    X = data['X']
    Y = data['Y']

    rows = []
    for compteur in range(n_sample):
        print("Point : ", compteur, "/", n_sample)
        try:  # si assez de données sur la zone choisie
            x, y = np.random.uniform(np.min(X) + cell_size, np.max(X) - cell_size), np.random.uniform(np.min(Y) + cell_size, np.max(Y) - cell_size)

            local_data_ref = local_cloud(x, y, cell_size, file_ref, data=data_ref)
            local_data, angle = local_cloud_with_angle(x, y, cell_size, file, data=data)

            xref, yref, zref = local_data_ref['X'].mean(),  local_data_ref['Y'].mean(), local_data_ref['Z'].mean()

            local_data_values = local_data.values
            A_lidar = np.c_[local_data_values[:, 0], local_data_values[:, 1], np.ones(local_data_values.shape[0])]
            Coeff, _, _, _ = scipy.linalg.lstsq(A_lidar, local_data_values[:, 2])
            z_lidar = Coeff[0] * xref + Coeff[1] * yref + Coeff[2]

            erreur_absolue = zref - z_lidar

            erreur_aleatoire = local_distance_to_plane(x, y, cell_size, file, data)[1]

            row = [angle, erreur_absolue, erreur_aleatoire]
            rows.append(row)


        except:
            pass

    df = pd.DataFrame(rows, columns=columns)



    # Scaling des données

    scaler = preprocessing.MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(data_scaled)
    df_scaled.columns = ['Angle', 'Erreur Absolue', 'Erreur Aleatoire']


    # Calcul de la PCA

    pca = PCA(n_components=3, random_state=1)
    pca.fit(df_scaled)

    df_reduced = pca.transform(df_scaled)
    df_reduced = pd.DataFrame(df_reduced, columns=['Dimension 1', 'Dimension 2', 'Dimensions 3'])

    pca_result = pca_results(df_scaled, pca)

    biplot(df_scaled, df_reduced, pca)


    # Création de clusters par Gaussian Mixure

    gmm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full').fit(df_reduced.values)

    plot_resultsGMM(df_reduced.values, gmm.predict(df_reduced.values), gmm.means_, gmm.covariances_, 0,
                    'Clustering des erreurs par Gaussian Mixture')




if __name__ == '__main__':
    main()
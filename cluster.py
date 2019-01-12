import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import pandas as pd


def load_data():
    """load the encoded cases"""
    with open("encoded_cases.pkl", "rb") as f:
        encoded_cases = pickle.load(f)
    return encoded_cases


def save_data(df):
    """save feature-reduced cases"""
    with open("reduced_cases.csv", "w") as f:
        df.to_csv(f)


def save_transforms(pca, kmeans):
    """save fitted pca and kmeans models"""
    with open("transforms.pkl", "wb") as f:
        pickle.dump((pca, kmeans), f)


def pca(encoding):
    """make PCA object, fit to data, return fitted pca and reduced data"""
    pca = PCA(n_components=40)
    fitted_pca = pca.fit(encoding)
    return (fitted_pca, list(pca.transform(encoding)))


def cluster(data):
    """clusters the data and returns the fitted clustering object and labels"""
    kmeans = KMeans(n_clusters=10, random_state=2).fit(list(data))
    return (kmeans, kmeans.labels_)


def main():
    """load the encoded cases, apply feature reduction and clustering"""

    """load data"""
    data = load_data()
    cases, encoding = [x[0] for x in data], np.array([x[1] for x in load_data()])
    df = pd.DataFrame()
    df['case'] = cases

    """apply feature reduction by PCA"""
    fitted_pca, df['reduced_data'] = pca(encoding)

    """apply k-means clustering"""
    kmeans, df['cluster'] = cluster(df['reduced_data'].values)

    """store reduced data in dataframe"""
    df = df.set_index('case')
    df.drop('reduced_data', axis=1, inplace=True)
    df = df.sort_values('cluster')
    print(df)

    """save the data and the models"""
    save_data(df)
    save_transforms(fitted_pca, kmeans)


if __name__ == "__main__":
    main()

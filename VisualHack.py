from typing import ValuesView
from numpy.lib.twodim_base import mask_indices
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.api as sm
import xgboost as xgb
import uncertainties
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import (LinearRegression,
                                  ElasticNet,
                                  Ridge,
                                  Lasso,
                                  BayesianRidge)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.covariance import EmpiricalCovariance
from sklearn import datasets
import plotly.graph_objects as go
import geopandas as gpd
import plotly.express as px
from urllib.request import urlopen
import json

# source email stuti.chakraborty@cmcvellore.ac.in

def data_anayl(stuff, first, second, check):
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')

    if check:
        df = df[["code"]]
        df = df.merge(stuff, how="left", left_on="code", right_on="State")
    else:
        df = df[["code", "state"]]
        df = df.merge(stuff, how="left", left_on="state", right_on="State")
    fig = go.Figure(data=go.Choropleth(
        locations=df['code'], # Spatial coordinates
        z = df['Value'].astype(float), # Data to be color-coded
        locationmode = 'USA-states', # set of locations match entries in `locations`
        colorscale = 'Viridis',
        colorbar_title = second,
    ))

    fig.update_layout(
        title_text = first,
        geo_scope='usa')
    fig.show()


def read_data(stuff, needed_data, variable):
    stuff = pd.read_csv(stuff)
    only_region = stuff[needed_data] == variable
    state_data = stuff[only_region]
    state_data = state_data[["State", "Value"]]
    state_data = state_data.groupby("State").sum()
    data_anayl(state_data, "2020 Number of Mental Health Care Actions by US State", "# of Reports", False)


def read_second_data(stuff):
    stuff = pd.read_csv(stuff)
    stuff = stuff[["State", "Incident ID"]]

    stuff = stuff.groupby("State").count()
    stuff = stuff.rename(columns={"Incident ID":"Value"})

    data_anayl(stuff, "2020 Number of Gun Related Incidents by US State", "# of Incidents", False)

def read_third_data(stuff):
    stuff = pd.read_csv(stuff)
    stuff = stuff.groupby("state")["positive"].sum().reset_index()
    stuff = stuff.rename(columns={"positive":"Value", "state":"State"})
    data_anayl(stuff, "2020/2021 Reported Covid Cases by US State", "# of Cases", True)


if __name__ == "__main__":
    read_data("/Users/walidsheykho/Desktop/Machine Learning/HackData/Mental_Health_Care_in_the_Last_4_Weeks.csv", "Group", "By State")
    read_second_data("/Users/walidsheykho/Desktop/Machine Learning/HackData/export-a8f28a63-447e-48de-adcd-4290f54be28b.csv")
    read_third_data("/Users/walidsheykho/Desktop/Machine Learning/HackData/all-states-history.csv")
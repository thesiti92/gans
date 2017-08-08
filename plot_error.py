import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py

df = pd.read_json(path_or_buf = "./result/log", orient = 'records')
gen = go.Scatter(
    x=df["epoch"],
    y=df["loss_gen"],
    mode="markers"
)
real = go.Scatter(
    x=df["epoch"],
    y=df["loss_data"],
    mode="markers"
)
layout = go.Layout(
    title='Error of Gan, zdim=1',
    xaxis={
        "title": 'Number of Epochs'
    },
    yaxis={
        "title": 'Error'
    }
)
fig = go.Figure(data=[gen, real], layout=layout)
plot_url = py.plot(fig, filename='error_scaled_z1')
print(plot_url)
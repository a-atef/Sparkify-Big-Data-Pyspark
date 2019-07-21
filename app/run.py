# import libraries
import re
import json
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from flask import Flask
from flask import render_template, request, jsonify

app = Flask(__name__)

# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
    """Generate Plots in the HTML index page.
    
        Args:
            None
            
        Returns:
            render_template(render_template): Render template for the plots  
        
    """
    # read data and create visuals for the original dataset
    df = read_data_csv("data/original_data.csv")

    # create table for original dataset
    table_1 = data_table(
        drop_cols=[
            "Unnamed: 0",
            "firstName",
            "lastName",
            "status",
            "userAgent",
            "length",
        ],
        parse_dates=["ts", "registration"],
        path="./data/original_data.csv",
    )

    # create and append plotly visuals into an array to be passed later for graphJSON file
    graphs = [
        table_1,
        check_nulls(df),
        counts_plot(df, "auth"),
        counts_plot(
            df,
            "location",
            x_axis="States",
            y_axis="Counts",
            title="Top 5 States",
            location=True,
        ),
        counts_plot(
            df, "page", x_axis="Page Type", y_axis="Counts", title="Page Distribution"
        ),
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


@app.route("/timeplot")
def timeplot():
    """Generate Plots in the HTML timeplot page.
    
        Args:
            None
            
        Returns:
            render_template(render_template): Render template for the plots  
        
    """
    # read data and create visuals
    df = read_data_csv("data/original_data.csv")

    # create and append plotly visuals into an array to be passed later for graphJSON file
    graphs = [new_customers_chart(df), hourly_log_chart(df)]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("timeplot.html", ids=ids, graphJSON=graphJSON)


@app.route("/distplot")
def distplot():
    """Generate Plots in the HTML distplot page.
    
        Args:
            None
            
        Returns:
            render_template(render_template): Render template for the plots  
        
    """
    # read data and create visuals
    df_thumbs_down = read_data_csv("./data/thumbsDown_data.csv")
    df_thumbs_up = read_data_csv("./data/thumbsUp_data.csv")
    df_days = read_data_csv("./data/days_data.csv")
    df_songsPlayed = read_data_csv("./data/songsPlayed_data.csv")

    graphs = [
        histogram(df_thumbs_up, "ThumbsUp"),
        histogram(df_thumbs_down, "ThumbsDown"),
        histogram(
            df_songsPlayed,
            "SongsPlayed",
            title="Histogram - Number of Songs Played",
            x_axis="Number of Songs Played",
        ),
        histogram(
            df_days,
            "Days",
            title="Histogram - Number of Days Users Stayed Loyal to Sparkify",
            x_axis="Number of Days",
        ),
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("distplot.html", ids=ids, graphJSON=graphJSON)


@app.route("/traindata")
def traindata():
    """Generate Plots in the traindata page.
    
        Args:
            None
            
        Returns:
            render_template(render_template): Render template for the plots  
        
    """
    # read data and create visuals
    df_features = read_data_csv("./data/features_data.csv")

    table_2 = data_table(
        drop_cols=["Unnamed: 0", "FeatureVector", "ScaledFeatures"],
        num_cols=["Days", "UpPerSong", "DownPerSong", "SongsPerHour"],
        title="Transformed Dataset - Sample Records",
    )

    graphs = [table_2, heat_map(df_features)]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("traindata.html", ids=ids, graphJSON=graphJSON)


@app.route("/results")
def results():
    """Generate Plots in the HTML master page.
    
        Args:
            None
            
        Returns:
            render_template(render_template): Render template for the plots  
        
    """
    # read data and create visuals
    df_f_measure = read_data_csv("data/f_measure_data.csv")
    df_pr = read_data_csv("data/pr_data.csv")
    df_roc = read_data_csv("data/roc_data.csv")

    graphs = [
        roc_plot(df_roc),
        recall_repcision_plot(df_pr),
        f_measure_plot(df_f_measure),
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("results.html", ids=ids, graphJSON=graphJSON)


def __capitalize__(word):
    """Split a camel case word into a sentence.
    
        Args:
            word (string): column name in camel case format
            
        Returns:
            string: return a clean sentence 
        
    """
    words = re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", word)).split()
    words = [word.capitalize() for word in words]
    return "-".join(words)


def data_table(
    drop_cols=[],
    num_cols=[],
    parse_dates=[],
    path="./data/features_data.csv",
    title="Original Dataset - Sample Records",
):
    """Create a table view using plotly.
    
        Args:
            drop_cols (list): columns to drop from the view
            num_cols (list): numerical columns to be rounded to the nearest 2 decimals 
            parse_dates (list): datetime column to be parsed as datetime columns
            path (string): path to read the data 
            title (string): table title
            
        Returns:
            Table (fig): a table view using plotly   
    """
    df = pd.read_csv(path, nrows=20)
    if len(drop_cols):
        df.drop(drop_cols, axis=1, inplace=True)

    if len(num_cols):
        df[num_cols] = df[num_cols].apply(lambda x: round(x, 2))
        df.columns = [__capitalize__(col) for col in df.columns]
    if len(parse_dates):
        for date in parse_dates:
            df[date] = df[date].apply(
                lambda x: datetime.datetime.fromtimestamp(x / 1000.0)
            )
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df.columns), fill_color="paleturquoise", align="center"
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    fill_color="lavender",
                    align="center",
                ),
            )
        ]
    )

    fig.update_layout(width=1300, title=go.layout.Title(text=title, x=0.5))

    return fig


def check_nulls(df):
    """Create a plot of the number of missing records in each column.
    
        Args:
            df (DataFrame): Dataframe object
            
        Returns:
            Figure (fig): a plotly figure  
    """

    nulls_df = df.isnull().sum()
    nulls_df = nulls_df[nulls_df > 0].sort_values(ascending=False)
    missing_records = nulls_df.tolist()
    column_names = nulls_df.index.tolist()

    x = [
        " ".join(re.sub("([a-z])([A-Z])", r"\1 \2", name).split()).capitalize()
        for name in column_names
    ]
    y = missing_records
    fig = go.Figure(
        [
            go.Bar(
                x=x,
                y=y,
                text=missing_records,
                marker_color="rgb(55, 83, 109)",
                textposition="auto",
                hovertext=[
                    "Missing Records in the `{}` column is {}".format(bar, y[i])
                    for i, bar in enumerate(x)
                ],
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title=go.layout.Title(text="Number of Missing Records/Column", x=0.5),
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Column Name")),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Counts")),
    )

    return fig


def counts_plot(
    df,
    col_name,
    x_axis="Authorization Type",
    y_axis="Counts",
    title="Authorization Type Counts",
    location=False,
):
    """Create a plot for a counts plot for a column.
    
        Args:
            df (DataFrame): Dataframe object
            col_name (string):  selected column for count plot 
            x_axis (string): x-axis_label
            y_axis (string): y-axis label
            title (string): plot title
            location (bool): If True returns a count plot for the top five locations in the location column
            
        Returns:
            Figure (fig): a plotly figure
        
    """
    x = []
    y = []

    if location:
        df = df[[col_name]]
        df.dropna(inplace=True)
        df["State"] = df[col_name].apply(lambda x: x.split(",")[1])
        x = [
            "California",
            "New York",
            "Texas",
            "Massachusetts",
            "Florida",
        ]  # df["State"].value_counts().index.tolist()[:5]
        y = df["State"].value_counts().tolist()[:5]
    else:
        df = df[col_name].value_counts()
        y = df.tolist()

        try:
            x = [
                " ".join(re.sub("([a-z])([A-Z])", r"\1 \2", name).split()).capitalize()
                for name in df.index.tolist()
            ]
        except TypeError:
            x = [
                "Male" if name == "M" else "Female"
                for name in df[col_name].unique().tolist()
            ]

    fig = go.Figure(
        [
            go.Bar(
                x=x,
                y=y,
                text=y,
                marker_color="rgb(55, 83, 109)",
                textposition="auto",
                hovertext=[
                    "Total counts of `{}` is {}".format(x[i], y[i])
                    for i, bar in enumerate(x)
                ],
                hoverinfo="text",
            )
        ]
    )
    fig.update_layout(
        title=go.layout.Title(text=title, x=0.5),
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=x_axis)),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text=y_axis)),
    )

    return fig


def new_customers_chart(df):
    """Create a timeline plot for the number of new customers enrolled each month.
    
        Args:
            df (DataFrame): Dataframe object
        Returns:
            Figure (fig): a plotly figure
        
    """

    registrations = df[["registration", "userId"]].dropna()
    registrations["date"] = pd.to_datetime(
        [
            datetime.datetime.fromtimestamp(date / 1000)
            for date in registrations.registration
        ]
    )
    series = registrations.resample("M", on="date").count()["registration"]
    fig = go.Figure()
    x = series.index.tolist()
    y = series.tolist()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            line=dict(color="firebrick", width=3),
            hovertext=[
                "Total number of new customers at `{}` is {}".format(x[i], y[i])
                for i, bar in enumerate(x)
            ],
            hoverinfo="text",
            mode="lines+markers",
            name="lines+markers",
            fill="toself",
            fillcolor="rgba(231,107,243,0.2)",
        )
    )
    fig.update_layout(
        title=go.layout.Title(text="Number of New Customers Each Month", x=0.5),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Counts")),
    )
    return fig


def hourly_log_chart(df):
    """Create a timeline plot for the number of new customers enrolled each month.
    
        Args:
            df (DataFrame): Dataframe object
        Returns:
            Figure (fig): a plotly figure
        
    """
    hourly_df = df[["userId", "page", "ts"]]
    hourly_df["hour"] = hourly_df["ts"].apply(
        (lambda x: datetime.datetime.fromtimestamp(x / 1000.0).hour)
    )
    hourly_df = hourly_df[hourly_df["page"] == "NextSong"].groupby("hour").count()
    fig = go.Figure()
    x = hourly_df.index.tolist()
    y = hourly_df.userId

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            line=dict(color="firebrick", width=3),
            hovertext=[
                "Number of Customers listening to songs at `{}:00` is {}".format(
                    x[i], y[i]
                )
                for i, bar in enumerate(x)
            ],
            hoverinfo="text",
            mode="lines+markers",
            name="lines+markers",
            fill="toself",
            fillcolor="rgba(231,107,243,0.2)",
        )
    )
    fig.update_layout(
        title=go.layout.Title(text="Hourly Log of Active Customers", x=0.5),
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Hour")),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Number of Customers")),
    )
    return fig


def histogram(
    df,
    column,
    title="Histogram - Number of Thumps Up",
    x_axis="Number of Thumbs Up",
    y_axis="Frequency",
):
    """Create a histogram for a column with the specified annotations.
    
        Args:
            df (DataFrame): Dataframe object
            column (string): the selected numerical column name
            title (string): plot title
            x_axis (string): x-axis_label
            y_axis (string): y-axis label
        Returns:
            Figure (fig): a plotly figure
    """
    series = df[column].tolist()
    fig = go.Figure(data=[go.Histogram(x=series, marker_color="rgb(55, 83, 109)")])
    fig.update_layout(
        title=go.layout.Title(text=title, x=0.5),
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=x_axis)),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text=y_axis)),
    )
    return fig


def heat_map(df):
    """Create a heatmap for the numerical features of the DataFrame.
    
        Args:
            df (DataFrame): Dataframe object
        Returns:
            Figure (fig): a plotly figurene
    """

    df_heat = df[["SongsPerHour", "SongsPlayed", "Days", "UpPerSong", "DownPerSong"]]
    df_heat = df_heat.corr(method="pearson")
    z = df_heat.values
    x = df_heat.columns.tolist()
    x = [__capitalize__(name) for name in x]
    font_colors = ["white"]
    colorscale = [[0, "navy"], [1, "plum"]]
    fig = ff.create_annotated_heatmap(
        z=z,
        x=x,
        y=x,
        annotation_text=np.around(z, decimals=2),
        colorscale=colorscale,
        font_colors=font_colors,
    )
    fig.update_layout(
        title=go.layout.Title(text="Heat Map | Numerical Features", x=0.5)
    )
    return fig


def roc_plot(df):
    """Create a timeline plot for the number of new customers enrolled each month.
    
        Args:
            df (DataFrame): Dataframe object
        Returns:
            Figure (fig): a plotly figure
        
    """
    x = df["FPR"].tolist()
    y = df["TPR"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            line=dict(color="firebrick", width=3),
            hoverinfo="text",
            mode="lines",
            name="lines",
        )
    )
    fig.update_layout(
        title=go.layout.Title(
            text=" Receiver Operating Characteristic Curve (ROC)", x=0.5
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(text="False Positive Rate (FPR)")
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(text=" True Positive Rate (TPR)")
        ),
    )
    return fig


def recall_repcision_plot(df):
    """Create a timeline plot for the number of new customers enrolled each month.
    
        Args:
            df (DataFrame): Dataframe object
        Returns:
            Figure (fig): a plotly figure
        
    """
    x = df["recall"].tolist()
    y = df["precision"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            line=dict(color="firebrick", width=3),
            hoverinfo="text",
            mode="lines",
            name="lines",
        )
    )
    fig.update_layout(
        title=go.layout.Title(text="Recall vs Precision Curve", x=0.5),
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Recall")),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Precision")),
    )
    return fig


def f_measure_plot(df):
    """Create a timeline plot for the number of new customers enrolled each month.
    
        Args:
            df (DataFrame): Dataframe object
        Returns:
            Figure (fig): a plotly figure
        
    """
    x = df["threshold"].tolist()
    y = df["F-Measure"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            line=dict(color="firebrick", width=3),
            hoverinfo="text",
            mode="lines",
            name="lines",
        )
    )
    fig.update_layout(
        title=go.layout.Title(text="Cutoff vs F-Score", x=0.5),
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Cuttoff")),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="F-Score")),
    )
    return fig


def read_data_csv(path):
    """Read the csv data from the specified path.
    
        Args:
            path (string): path to the csv file
        Returns:
            DataFrame (df): DataFrame Object
        
    """
    df = pd.read_csv(path)
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    return df


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()

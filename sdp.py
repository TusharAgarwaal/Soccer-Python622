import matplotlib.pyplot as plt
from shiny import App, render, ui
import pandas as pd

# Load the FIFA dataset
file_path = 'C:/Users/tusha/Downloads/fifa21_male2.csv'
fifa_data = pd.read_csv(file_path, low_memory=False)

# Clean the dataset
def clean_data(data):
    data.fillna({
        'Position': 'Unknown',
        'OVA': data['OVA'].median(),
        'POT': data['POT'].median(),
    }, inplace=True)
    # Normalize text columns
    data['Position'] = data['Position'].str.lower().str.strip()
    data['Nationality'] = data['Nationality'].str.lower().str.strip()
    data['Club'] = data['Club'].str.lower().str.strip()
    return data

fifa_data = clean_data(fifa_data)

# Define UI
app_ui = ui.page_fluid(
    ui.h2("Scrollable Bar Chart for FIFA Player Analysis"),
    ui.input_select("x_axis", "Select X-axis", choices=["Position", "Nationality", "Club"], selected="Club"),
    ui.input_select("y_axis", "Select Y-axis", choices=["OVA", "POT", "Age", "Strength"], selected="OVA"),
    ui.input_slider("top_n", "Show Top N Categories", min=5, max=50, value=10, step=1),
    ui.output_plot("bar_chart")
)

# Define Server Logic
def server(input, output, session):
    @output
    @render.plot
    def bar_chart():
        # Get user inputs
        x_axis = input.x_axis()
        y_axis = input.y_axis()
        top_n = input.top_n()

        # Group and aggregate the data based on the user selection
        grouped_data = fifa_data.groupby(x_axis)[y_axis].mean().sort_values(ascending=False).head(top_n)

        # Adjust figure size dynamically based on top N
        fig_width = max(10, top_n * 0.5)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        # Create a bar chart
        grouped_data.plot(kind="bar", color="skyblue", edgecolor="black", ax=ax)
        ax.set_title(f"Top {top_n} Average {y_axis} by {x_axis.capitalize()}")
        ax.set_xlabel(x_axis.capitalize())
        ax.set_ylabel(f"Average {y_axis}")
        plt.xticks(rotation=45, fontsize=10)  # Rotate labels with smaller font
        plt.tight_layout()

        return fig

# Create and run the Shiny app
app = App(app_ui, server)
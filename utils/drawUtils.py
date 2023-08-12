import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def draw_bar_plot(data, save_path,title):
    data_0 = [x[0] for x in data if x[1] == 0]
    data_1 = [x[0] for x in data if x[1] == 1]
    plt.boxplot([data_0, data_1], vert=False, patch_artist=True,
                boxprops=dict(facecolor='red', color='red'),
                whiskerprops=dict(color='red'),
                capprops=dict(color='red'),
                medianprops=dict(color='white'),
                flierprops=dict(markerfacecolor='red', markeredgecolor='red'),
                )
    plt.yticks([1, 2], ['True', 'False'])
    plt.xlabel('similarity scores')
    plt.title(f"Boxplot of {title} Data")
    plt.savefig(save_path)
    plt.show()

def draw_hist(data, save_path,title):
    data_0 = [x[0] for x in data if x[1] == 0]
    data_1 = [x[0] for x in data if x[1] == 1]
    plt.hist([data_0, data_1], bins=30, color=['red', 'blue'], label=['False', 'True'])
    plt.xlabel('similarity scores')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title("Histogram of Third Stage Data")
    plt.savefig(save_path)
    plt.show()
    

def draw_scatter_plot(data, save_path, title):
    # Splitting the data based on the second dimension
    data_0 = [x[0] for x in data if x[1] == 0]
    data_1 = [x[0] for x in data if x[1] == 1]

    # Using a constant for y to keep all points on the same level
    y_0 = [0]*len(data_0)
    y_1 = [0]*len(data_1)

    # Plotting
    plt.scatter(data_0, y_0, color='red', label='False')
    plt.scatter(data_1, y_1, color='blue', label='True')
    plt.xlabel('similarity scores')
    plt.yticks([])  # Hide y-axis
    plt.legend()
    plt.title(f"{title} Data Visualization")
    plt.savefig(save_path)
    plt.show()
    

def draw_swarm_plot(data, save_path, title):
    # Convert data to DataFrame for seaborn
    df = pd.DataFrame(data[:1000], columns=['similarity scores', 'Category'])

    # Create a swarm plot
    sns.swarmplot(x='Category', y='similarity scores', data=df)

    # Setting the labels and title
    plt.xlabel('similarity scores')
    plt.ylabel('Value')
    plt.title(f"Swarmplot of {title} Data")

    # Show the plot
    plt.savefig(save_path)
    plt.show()

def draw_all(data, save_path, title):
    draw_bar_plot(data, save_path + 'bar_plot.png',title)
    draw_hist(data, save_path + 'hist.png',title)
    draw_scatter_plot(data, save_path + 'scatter_plot.png',title)
    draw_swarm_plot(data, save_path + 'swarm_plot.png',title)
    
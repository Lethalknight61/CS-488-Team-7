import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Hero Name Data
heroes_data = {
    1: "Anti-Mage",
    2: "Axe",
    3: "Bane",
    4: "Bloodseeker",
    5: "Crystal Maiden",
    6: "Drow Ranger",
    7: "Earthshaker",
    8: "Juggernaut",
    9: "Mirana",
    10: "Morphling",
    11: "Shadow Fiend",
    12: "Phantom Lancer",
    13: "Puck",
    14: "Pudge",
    15: "Razor",
    16: "Sand King",
    17: "Storm Spirit",
    18: "Sven",
    19: "Tiny",
    20: "Vengeful Spirit",
    21: "Windranger",
    22: "Zeus",
    23: "Kunkka",
    25: "Lina",
    26: "Lion",
    27: "Shadow Shaman",
    28: "Slardar",
    29: "Tidehunter",
    30: "Witch Doctor",
    31: "Lich",
    32: "Riki",
    33: "Enigma",
    34: "Tinker",
    35: "Sniper",
    36: "Necrophos",
    37: "Warlock",
    38: "Beastmaster",
    39: "Queen of Pain",
    40: "Venomancer",
    41: "Faceless Void",
    42: "Skeleton King",
    43: "Death Prophet",
    44: "Phantom Assassin",
    45: "Pugna",
    46: "Templar Assassin",
    47: "Viper",
    48: "Luna",
    49: "Dragon Knight",
    50: "Dazzle",
    51: "Clockwerk",
    52: "Leshrac",
    53: "Nature's Prophet",
    54: "Lifestealer",
    55: "Dark Seer",
    56: "Clinkz",
    57: "Omniknight",
    58: "Enchantress",
    59: "Huskar",
    60: "Night Stalker",
    61: "Broodmother",
    62: "Bounty Hunter",
    63: "Weaver",
    64: "Jakiro",
    65: "Batrider",
    66: "Chen",
    67: "Spectre",
    68: "Ancient Apparition",
    69: "Doom",
    70: "Ursa",
    71: "Spirit Breaker",
    72: "Gyrocopter",
    73: "Alchemist",
    74: "Invoker",
    75: "Silencer",
    76: "Outworld Devourer",
    77: "Lycanthrope",
    78: "Brewmaster",
    79: "Shadow Demon",
    80: "Lone Druid",
    81: "Chaos Knight",
    82: "Meepo",
    83: "Treant Protector",
    84: "Ogre Magi",
    85: "Undying",
    86: "Rubick",
    87: "Disruptor",
    88: "Nyx Assassin",
    89: "Naga Siren",
    90: "Keeper of the Light",
    91: "Wisp",
    92: "Visage",
    93: "Slark",
    94: "Medusa",
    95: "Troll Warlord",
    96: "Centaur Warrunner",
    97: "Magnus",
    98: "Timbersaw",
    99: "Bristleback",
    100: "Tusk",
    101: "Skywrath Mage",
    102: "Abaddon",
    103: "Elder Titan",
    104: "Legion Commander",
    105: "Techies",
    106: "Ember Spirit",
    107: "Earth Spirit",
    108: "Abyssal Underlord",
    109: "Terrorblade",
    110: "Phoenix",
    111: "Oracle",
    112: "Winter Wyvern",
    113: "Arc Warden"
}

heroNames = list(heroes_data.values())
heroKeys = list(heroes_data.keys())

# Function to parse a single row
def parse_row(row):
    # Extracting the features
    team_won = int(row[0])
    cluster_id = int(row[1])
    game_mode = int(row[2])
    game_type = int(row[3])
    heroes = [int(hero) for hero in row[4:]]
    
    team1_heroes = []
    team2_heroes = []

    for i in range(len(heroes)-1):
        if heroes[i] == -1:
            team1_heroes.append(heroNames[i])
        elif heroes[i] == 1:
            team2_heroes.append(heroNames[i])
    
    # Returning the parsed data
    return {
        'team_won': team_won,
        'cluster_id': cluster_id,
        'game_mode': game_mode,
        'game_type': game_type,
        'team1_heroes': team1_heroes,
        'team2_heroes': team2_heroes,
        'heroes': heroes
    }

# Path to the CSV file
csv_file_path = 'dota2Train.csv'

# List to store parsed data
parsed_data = []

# Reading the CSV file
with open(csv_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        parsed_data.append(parse_row(row))

# Extracting hero data from the dataset
heroes_data = [d['heroes'] for d in parsed_data]

# Convert heroes data into numpy array
heroes_array = np.array(heroes_data)

# Number of clusters
k = 4

# Perform k-means clustering
kmeans = KMeans(n_clusters=k)
kmeans.fit(heroes_array)

# Get the cluster centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Print the centroids and labels
print("Centroids:")
print(centroids)
print("\nLabels:")
print(labels)

# cluster labels 
cluster_labels = np.array(labels)

# match outcomes 
match_outcomes = np.array([match['team_won'] for match in parsed_data])

# Number of clusters (assuming clusters are labeled as integers)
num_clusters = len(np.unique(cluster_labels))

# Initialize dictionaries to store counts of winning and losing matches in each cluster
win_count = {}
loss_count = {}

# Iterate through each cluster
for cluster_label in range(num_clusters):
    # Get the indices of matches in the current cluster
    cluster_indices = np.where(cluster_labels == cluster_label)[0]
    
    # Count the number of winning and losing matches in the current cluster
    win_count[cluster_label] = np.sum(match_outcomes[cluster_indices] == 1)
    loss_count[cluster_label] = np.sum(match_outcomes[cluster_indices] == 0)

# Calculate proportions of winning and losing matches in each cluster
total_matches = len(match_outcomes)
win_proportions = {cluster_label: count / total_matches for cluster_label, count in win_count.items()}
loss_proportions = {cluster_label: count / total_matches for cluster_label, count in loss_count.items()}

# Visualize the outcome distribution
fig, ax = plt.subplots(figsize=(8, 6))
clusters = np.arange(num_clusters)
bar_width = 0.35
opacity = 0.8

# Bar plot for winning matches
ax.bar(clusters, list(win_proportions.values()), bar_width, alpha=opacity, color='b', label='Winning Matches')

# Bar plot for losing matches
ax.bar(clusters + bar_width, list(loss_proportions.values()), bar_width, alpha=opacity, color='r', label='Losing Matches')

ax.set_xlabel('Cluster')
ax.set_ylabel('Proportion')
ax.set_title('Outcome Distribution in Clusters')
ax.set_xticks(clusters + bar_width / 2)
ax.set_xticklabels(clusters)
ax.legend()

plt.tight_layout()
plt.show()

# Assuming you have already performed clustering and have cluster labels
# Replace these with your actual cluster labels and data
  # Example cluster labels (0 and 1)
numerical_features = np.array([match['heroes'] for match in parsed_data])
# Perform dimensionality reduction using PCA
pca = PCA(n_components=2)
data_2d = pca.fit_transform(numerical_features)

# Plot data points colored by cluster
plt.figure(figsize=(8, 6))
for cluster_label in np.unique(cluster_labels):
    plt.scatter(data_2d[cluster_labels == cluster_label, 0],
                data_2d[cluster_labels == cluster_label, 1],
                label=f'Cluster {cluster_label}')

# Optionally, plot cluster centroids
centroid_2d = pca.transform(centroids)  # If you have cluster centroids
plt.scatter(centroid_2d[:, 0], centroid_2d[:, 1], marker='X', color='black', s=100, label='Centroids')

plt.title('Scatter Plot with Clustering Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

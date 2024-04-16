import csv
import json

# Load hero data from JSON
with open('dota2Train.json') as f:
    heroes_data = json.load(f)

# Create a dictionary mapping hero IDs to their names
heroes_map = {hero['id']: hero['localized_name'] for hero in heroes_data['heroes']}

# Function to parse a single row
def parse_row(row):
    # Extracting the features
    team_won = int(row[0])
    cluster_id = int(row[1])
    game_mode = int(row[2])
    game_type = int(row[3])
    team1_heroes = []
    team2_heroes = []

    # Loop through the hero IDs in the row
    for hero_id in row[4:]:
        hero_id = int(hero_id)
        hero_name = heroes_map[abs(hero_id)]  # Get hero name
        if hero_id > 0:
            # Map hero ID to hero name for Team 1
            team1_heroes.append(hero_name)
        elif hero_id < 0:
            # Map hero ID to hero name for Team 2
            team2_heroes.append(hero_name)

    # Returning the parsed data
    return {
        'team_won': team_won,
        'cluster_id': cluster_id,
        'game_mode': game_mode,
        'game_type': game_type,
        'team1_heroes': team1_heroes,
        'team2_heroes': team2_heroes
    }

# Path to the CSV file
csv_file_path = 'dota2Test.csv'

# List to store parsed data
parsed_data = []

# Reading the CSV file
with open(csv_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        parsed_data.append(parse_row(row))

# Example usage: Print the parsed data of the first row
print(parsed_data[0])

from createCSV import augment_csv_data

# Double toutes les données
augment_csv_data("dataset_landmarks.csv")

# Est censé équilibrer les classes à un nombre cible.
# bugué pour l'instant car il double aussi toutes les autres classes
# TODO
# augment_csv_data(
#     "dataset_landmarks.csv",
#     target_samples={"Ellie": 80, "AngryEmoji": 80},
#     noise_level=0.02
# )

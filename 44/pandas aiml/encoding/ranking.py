import pandas as pd 

data = {
    "Name": [
        "Rohan", "Priya", "Amit", "Sana", "Vikram", "Sneha", "Rahul", "Meena", "Alok", "Kavya",
        "Dinesh", "Jaya", "Ritu", "Deepak", "Isha", "Guru", "Pooja", "Anil", "Akash", "Tara"
    ],

    "PreferredMode": [
        "Online", "Offline", "Hybrid", "Online", "Offline", "Online", "Hybrid", "Online", "Offline", "Online",
        "Hybrid", "Offline", "Online", "Hybrid", "Online", "Offline", "Hybrid", "Offline", "Online", "Hybrid"
    ],
    "Marks": [
        85, 79, 90, 82, 75, 88, 79, 91, 76, 84,
        89, 78, 83, 85, 87, 74, 84, 73, 86, 70
    ]
}

df = pd.DataFrame(data)
print(df.info())

df["rank_min"] = df['Marks'].rank(method='min')
print(df.sort_values("Marks"))


df["rank_max"] = df['Marks'].rank(method='max')
print(df.sort_values("Marks"))



df["rank_first"] = df['Marks'].rank(method='first')
print(df.sort_values("Marks"))



df["rank_average"] = df['Marks'].rank(method='average')
print(df.sort_values("Marks"))


df["rank_dense      "] = df['Marks'].rank(method='dense')
print(df.sort_values("Marks"))





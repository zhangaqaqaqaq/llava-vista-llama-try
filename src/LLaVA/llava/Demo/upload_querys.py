import json
import mysql.connector


# Load JSON file
json_path = r'D:\Projects\LLaVA\datas\querys\coco_pope_adversarial.json'
with open(json_path, 'r') as file:
    data = json.load(file)

# Connect to the MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="user",
    password="abc123",
    database="MyDataBase"
)

cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS database_json (
        question_id INT PRIMARY KEY,
        image VARCHAR(255),
        text TEXT,
        label VARCHAR(255)
    )
''')

# Insert JSON data into the table
for item in data:
    cursor.execute('''
        INSERT INTO database_json (question_id, image, text, label)
        VALUES (%s, %s, %s, %s)
    ''', (item['question_id'], item['image'], item['text'], item['label']))

# Commit the transaction
conn.commit()

# Close the connection
conn.close()
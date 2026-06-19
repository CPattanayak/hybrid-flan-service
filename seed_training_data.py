"""
Run this ONCE before training to populate the training_data table.
  python seed_training_data.py
"""
import psycopg2

DB_URL = "postgresql://postgres:password@localhost:5432/sql_training_db"

conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

# Optional: wipe old rows so we start clean
cur.execute("DELETE FROM training_data")

training_data = []

# --- SELECT patterns ---
for i in range(1, 51):
    training_data.append((
        f"Show all orders for customer {i}",
        f"SELECT * FROM orders WHERE customer_id={i};"
    ))
    training_data.append((
        f"Get all orders above amount {i * 10}",
        f"SELECT * FROM orders WHERE amount>{i * 10};"
    ))
    month = str((i % 9) + 1).zfill(2)
    training_data.append((
        f"List orders placed after 2024-{month}-01",
        f"SELECT * FROM orders WHERE order_date > '2024-{month}-01';"
    ))

# --- SELECT all ---
training_data += [
    ("Show all customers",           "SELECT * FROM customers;"),
    ("List all orders",              "SELECT * FROM orders;"),
    ("Get every customer record",    "SELECT * FROM customers;"),
    ("Fetch all order records",      "SELECT * FROM orders;"),
    ("Show me every customer",       "SELECT * FROM customers;"),
    ("Display all orders",           "SELECT * FROM orders;"),
]

# --- INSERT patterns ---
names = ["Alice", "Bob", "Carol", "Dan", "Eva",
         "Frank", "Grace", "Hana", "Ivan", "Judy",
         "Karl", "Luna", "Mike", "Nina", "Omar",
         "Priya", "Quinn", "Rosa", "Sam", "Tara"]
for i, name in enumerate(names * 5):
    email = f"{name.lower()}{i}@example.com"
    training_data.append((
        f"Add a new customer named {name} with email {email}",
        f"INSERT INTO customers (name, email) VALUES ('{name}', '{email}');"
    ))
    training_data.append((
        f"Create customer {name}, email {email}",
        f"INSERT INTO customers (name, email) VALUES ('{name}', '{email}');"
    ))
    training_data.append((
        f"Insert new customer: {name}, {email}",
        f"INSERT INTO customers (name, email) VALUES ('{name}', '{email}');"
    ))

# --- INSERT orders ---
for i in range(1, 31):
    amount = i * 15
    training_data.append((
        f"Add order for customer {i} with amount {amount}",
        f"INSERT INTO orders (customer_id, amount, order_date) VALUES ({i}, {amount}, CURRENT_DATE);"
    ))

# --- UPDATE patterns ---
for i in range(1, 51):
    email = f"user{i}@newmail.com"
    training_data.append((
        f"Update customer {i}'s email to {email}",
        f"UPDATE customers SET email='{email}' WHERE id={i};"
    ))
    training_data.append((
        f"Change email of customer {i} to {email}",
        f"UPDATE customers SET email='{email}' WHERE id={i};"
    ))

# --- UPDATE order amount ---
for i in range(1, 21):
    amount = i * 20
    training_data.append((
        f"Set order {i} amount to {amount}",
        f"UPDATE orders SET amount={amount} WHERE id={i};"
    ))

# --- DELETE patterns ---
for i in range(1, 51):
    training_data.append((
        f"Delete order {i}",
        f"DELETE FROM orders WHERE id={i};"
    ))
    training_data.append((
        f"Remove order {i}",
        f"DELETE FROM orders WHERE id={i};"
    ))

for i in range(1, 21):
    training_data.append((
        f"Delete customer {i}",
        f"DELETE FROM customers WHERE id={i};"
    ))
    training_data.append((
        f"Remove customer {i}",
        f"DELETE FROM customers WHERE id={i};"
    ))

# --- JOIN patterns ---
for i in range(1, 31):
    training_data.append((
        f"Show customer names and their order amounts for customer {i}",
        f"SELECT c.name, o.amount FROM customers c JOIN orders o ON c.id=o.customer_id WHERE c.id={i};"
    ))
    training_data.append((
        f"Get name and order total for customer {i}",
        f"SELECT c.name, o.amount FROM customers c JOIN orders o ON c.id=o.customer_id WHERE c.id={i};"
    ))

# --- Aggregate patterns ---
training_data += [
    ("Count all customers",
     "SELECT COUNT(*) FROM customers;"),
    ("How many orders are there",
     "SELECT COUNT(*) FROM orders;"),
    ("Total order amount for all orders",
     "SELECT SUM(amount) FROM orders;"),
    ("Average order amount",
     "SELECT AVG(amount) FROM orders;"),
    ("Highest order amount",
     "SELECT MAX(amount) FROM orders;"),
    ("Lowest order amount",
     "SELECT MIN(amount) FROM orders;"),
    ("Total amount spent by each customer",
     "SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id;"),
    ("Number of orders per customer",
     "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id;"),
]

# --- ORDER BY / LIMIT patterns ---
training_data += [
    ("Show top 5 orders by amount",
     "SELECT * FROM orders ORDER BY amount DESC LIMIT 5;"),
    ("Get 10 most recent orders",
     "SELECT * FROM orders ORDER BY order_date DESC LIMIT 10;"),
    ("List customers alphabetically",
     "SELECT * FROM customers ORDER BY name ASC;"),
    ("Show the latest order",
     "SELECT * FROM orders ORDER BY order_date DESC LIMIT 1;"),
]

# Insert all rows
cur.executemany(
    "INSERT INTO training_data (input_text, target_sql) VALUES (%s, %s)",
    training_data
)
conn.commit()
cur.close()
conn.close()
print(f"✅ Inserted {len(training_data)} training rows into training_data table.")

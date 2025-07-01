from pymongo import MongoClient
from dotenv import load_dotenv
import os
import random

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["po_management"]
collection = db["purchase_orders"]

# Status options
status_options = ["In Progress", "Completed"]
payment_status_options = ["Pending", "Paid"]
possible_flags = [
    "Price mismatch",
    "Missing invoice approval",
    "Custom branding not confirmed",
    "Late delivery risk"
]

# Fetch and update each PO
for po in collection.find():
    updates = {}

    # Add status if missing
    if "status" not in po:
        updates["status"] = random.choice(status_options)

    # Add payment_status if missing
    if "payment_status" not in po:
        updates["payment_status"] = random.choice(payment_status_options)

    # Add red_flags randomly if not already present
    if "red_flags" not in po:
        updates["red_flags"] = random.sample(possible_flags, k=random.randint(0, 2))

    # Update document if changes found
    if updates:
        collection.update_one({"_id": po["_id"]}, {"$set": updates})
        print(f"Updated PO {po['po_number']}")

print("All POs updated successfully.")

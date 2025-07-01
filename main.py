from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile, Form
import fitz 
from pydantic import BaseModel
from typing import List
from difflib import get_close_matches
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import openai
import os
import json

# Load .env variables and override existing ones
load_dotenv(override=True)

openai.api_key = os.getenv("OPENAI_API_KEY")
mongo_uri = os.getenv("MONGO_URI")
mongo_client = MongoClient(mongo_uri)
db = mongo_client["po_management"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility to convert ObjectId to str recursively
def clean_ids(obj):
    if isinstance(obj, list):
        return [clean_ids(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: clean_ids(v) for k, v in obj.items()}
    elif isinstance(obj, ObjectId):
        return str(obj)
    else:
        return obj

# Pydantic models
class LogAction(BaseModel):
    po_number: str
    action: str
    actor: str
    approved: bool

class InvoiceRequest(BaseModel):
    po_number: str

class SendEmailRequest(BaseModel):
    po_number: str
    invoice: dict

class InvoiceData(BaseModel):
    po_number: str
    bill_to: str
    vendor: str
    total: float
    items: list

class SendInvoiceRequest(BaseModel):
    recipient: str
    invoice: InvoiceData

class ItemUpdate(BaseModel):
    description: str
    qty: int

class ConfirmUpdateRequest(BaseModel):
    po_number: str
    item_updates: List[ItemUpdate]

# Red flag helper
def check_and_create_red_flag_notification(po):
    if "red_flags" in po and po["red_flags"]:
        for flag in po["red_flags"]:
            if isinstance(flag, dict):
                title = f"Red Flag: {flag.get('title', 'Unknown')}"
                message = flag.get("description", "No description.")
            else:
                title = f"Red Flag: {flag}"
                message = flag
            db["notifications"].update_one(
                {"po_number": po["po_number"], "title": title},
                {"$setOnInsert": {
                    "type": "red_flag",
                    "message": message,
                    "action_required": True,
                    "created_at": datetime.utcnow(),
                    "status": "unread"
                }},
                upsert=True
            )

@app.get("/po-list")
def get_po_list():
    collection = db["purchase_orders"]
    results = list(collection.find({}, {
        "_id": 0,
        "po_number": 1,
        "date": 1,
        "bill_to.company": 1,
        "vendor.company": 1,
        "summary.total": 1,
        "status": 1,
        "payment_status": 1,
        "red_flags": 1
    }))
    for po in results:
        po["bill_to"] = po.get("bill_to", {}).get("company", "N/A")
        po["vendor"] = po.get("vendor", {}).get("company", "N/A")
    return results

@app.get("/po/{po_number}")
def get_po_detail(po_number: str):
    po = db["purchase_orders"].find_one({"po_number": po_number})
    emails = list(db["emails"].find({"po_number": po_number}))
    if not po:
        raise HTTPException(status_code=404, detail="PO not found")
    check_and_create_red_flag_notification(po)
    return {
        "po": clean_ids(po),
        "emails": clean_ids(emails)
    }

@app.get("/metrics")
def get_metrics():
    col = db["purchase_orders"]
    return {
        "total_orders": col.count_documents({}),
        "in_progress": col.count_documents({"status": "In Progress"}),
        "completed": col.count_documents({"status": "Completed"}),
        "payment_pending": col.count_documents({"payment_status": "Pending"})
    }

@app.get("/notifications")
def get_notifications():
    return list(db["notifications"].find({}, {"_id": 0}))

@app.post("/notifications/mark-read")
def mark_notification_read(po_number: str = Body(...), title: str = Body(...)):
    result = db["notifications"].update_many(
        {"po_number": po_number, "title": title},
        {"$set": {"status": "read"}}
    )
    return {"updated": result.modified_count}

@app.post("/log-action")
def log_user_action(log: LogAction):
    db["user_actions"].insert_one(log.dict())
    return {"status": "logged"}


@app.post("/ai/invoice")
def generate_invoice(req: InvoiceRequest):
    po = db["purchase_orders"].find_one({"po_number": req.po_number})
    emails = list(db["emails"].find({"po_number": req.po_number}))
    
    if not po:
        raise HTTPException(status_code=404, detail="PO not found")

    po_clean = clean_ids(po)

    # Direct structured invoice generation from PO
    invoice = {
        "po_number": po_clean["po_number"],
        "bill_to": po_clean.get("bill_to", {}),
        "vendor": po_clean.get("vendor", {}),
        "items": [],
        "summary": po_clean.get("summary", {}),
        "notes": po_clean.get("notes", "")
    }

    # Process items with safe casting to avoid NaN
    for item in po_clean.get("items", []):
        invoice_item = {
            "description": item.get("description"),
            "qty": int(item.get("qty", 0)),
            "unit_price": float(item.get("unit_price", 0.0)),
            "total": float(item.get("total", item.get("qty", 0) * item.get("unit_price", 0.0)))
        }
        invoice["items"].append(invoice_item)

    # Optional: Use AI for enrichment ONLY if needed
    # If you wish to keep AI formatting, use below code:

    prompt = f"""
    You are an invoice generation AI. Convert the following PO data into an invoice JSON while preserving:
    - Exact quantities, unit prices, totals
    - All summary values (subtotal, discount, tax, freight, total) exactly as provided
    - No recalculation, only structured reformatting
    PO Data: {json.dumps(invoice)}
    Return a valid JSON object.
    """

    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        invoice = json.loads(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI parsing failed: {str(e)}")

    # Insert directly into invoices collection for records
    db["invoices"].insert_one({"po_number": req.po_number, "invoice": invoice})

    return invoice


@app.post("/ai/send-invoice")
def send_invoice(req: SendInvoiceRequest):
    invoice_data = req.invoice.dict()
    recipient = req.recipient
    po_number = invoice_data.get("po_number")

    if not invoice_data or not po_number:
        raise HTTPException(status_code=400, detail="Missing invoice data or po_number.")

    db["email_logs"].insert_one({
        "po_number": po_number,
        "recipient": recipient,
        "subject": f"Invoice for {po_number}",
        "body": f"Attached is the invoice for {po_number}.",
        "invoice_data": invoice_data,
        "status": "sent",
        "timestamp": datetime.utcnow()
    })

    return {"success": True, "message": "Invoice email simulated and logged."}

@app.post("/ai/send-reminder")
def send_reminder(req: InvoiceRequest):
    print(f"Sending reminder for {req.po_number} to client...")
    return {"status": "reminder_sent"}


@app.post("/ai/update-po-from-emails")
def update_po_from_emails(req: InvoiceRequest):
    po = db["purchase_orders"].find_one({"po_number": req.po_number})
    emails = list(db["emails"].find({"po_number": req.po_number}))

    if not po or not emails:
        raise HTTPException(status_code=404, detail="PO or emails not found")

    # Sort emails chronologically for clear summaries
    emails.sort(key=lambda e: e.get("date", ""), reverse=False)

    po_items = po.get("items", [])
    item_context = "\n".join(
        [f"- {item['description']} (qty: {item['qty']}, unit_price: {item['unit_price']})" for item in po_items]
    )

    prompt = f"""
You are a procurement assistant AI helping to update a purchase order (PO) based on client email threads.

Instructions:
1. Read the email thread below carefully.
2. Match item names partially to the closest PO item.
3. If a quantity update is mentioned, infer the matching item name and calculate total_price = quantity x unit_price using the PO’s original price.
4. Detect any red flags like mismatches, disputes, missing details.
5. Generate a **short pointer summary** of the email thread capturing actionable decisions, requests, or confirmations in clear bullet points.

Respond in the following JSON format:

{{
"updates": {{
    "Item Name (from PO)": {{
    "quantity": <int>,
    "unit_price": <float>,
    "total_price": <float>
    }}
}},
"red_flags": [
    {{
    "message": "short description of issue",
    "ai_suggestion": "how to fix it"
    }}
],
"email_summary": [
    "First pointer summary line.",
    "Second pointer summary line."
]
}}

Email thread:
{json.dumps(clean_ids(emails))}
"""

    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        ai_result = json.loads(response.choices[0].message.content)

        # Extract updates, red flags, and summary
        raw_updates = ai_result.get("updates", {})
        updates = {}
        summary = ai_result.get("email_summary", [])
        red_flags = ai_result.get("red_flags", [])

        # Compare updates to existing PO items
        for name, changes in raw_updates.items():
            for item in po.get("items", []):
                if name.lower() in item["description"].lower():
                    same_qty = changes.get("quantity") == item.get("qty")
                    same_price = round(changes.get("unit_price", 0), 2) == round(item.get("unit_price", 0), 2)
                    if not (same_qty and same_price):
                        updates[item["description"]] = {
                            "quantity": changes.get("quantity"),
                            "unit_price": round(changes.get("unit_price", 0), 2),
                            "total_price": round(changes.get("quantity", 0) * changes.get("unit_price", 0), 2)
                        }
                    break

        # Save updates if any
        if updates:
            db["purchase_orders"].update_one(
                {"po_number": req.po_number},
                {"$set": {"updates": updates}}
            )
            db["user_actions"].insert_one({
                "po_number": req.po_number,
                "action": "AI PO Update",
                "actor": "AI Agent",
                "approved": False,
                "timestamp": datetime.utcnow(),
                "details": updates
            })

        # Save summary in PO document
        if summary:
            db["purchase_orders"].update_one(
                {"po_number": req.po_number},
                {"$set": {"email_summary": summary}}
            )

        #  Handle red flags
        new_flags = []
        for flag in red_flags:
            msg = flag.get("message")
            suggestion = flag.get("ai_suggestion")
            if msg:
                new_flags.append(msg)
                db["notifications"].insert_one({
                    "po_number": req.po_number,
                    "type": "Red Flag",
                    "message": msg,
                    "ai_suggestion": suggestion or "No suggestion",
                    "timestamp": datetime.utcnow()
                })

        # Return response
        return {
            "po_number": req.po_number,
            "updates": updates,
            "red_flags": new_flags,
            "email_summary": summary,
            "status": "PO updated via AI" if updates or new_flags else "No updates applied"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI parsing failed: {str(e)}")


@app.post("/po/confirm-update")
def confirm_update(po_number: str = Body(...), item_updates: list = Body(...)):
    po = db["purchase_orders"].find_one({"po_number": po_number})
    if not po:
        raise HTTPException(status_code=404, detail="PO not found")

    existing_items = po.get("items", [])
    item_dict = {item["description"]: item for item in existing_items}
    all_descriptions = list(item_dict.keys())

    # Fuzzy match item descriptions if exact key not found
    for update in item_updates:
        raw_desc = update["description"]
        matched_desc = raw_desc

        if raw_desc not in item_dict:
            closest = get_close_matches(raw_desc, all_descriptions, n=1, cutoff=0.5)
            if closest:
                matched_desc = closest[0]
            else:
                raise HTTPException(status_code=400, detail=f"No close match found for item '{raw_desc}'")

        if matched_desc in item_dict:
            item_dict[matched_desc]["qty"] = update["qty"]
            item_dict[matched_desc]["total"] = round(update["qty"] * item_dict[matched_desc]["unit_price"], 2)

    updated_items = list(item_dict.values())

    # Recalculate Summary
    subtotal = sum(item["qty"] * item["unit_price"] for item in updated_items)
    total_qty = sum(item["qty"] for item in updated_items)
    discount = round(0.03 * subtotal, 2)
    freight = round(10 + 0.05 * total_qty, 2)
    tax = round(0.10 * (subtotal - discount + freight), 2)
    total = round(subtotal - discount + freight + tax, 2)

    updated_summary = {
        "subtotal": round(subtotal, 2),
        "discount": discount,
        "freight": freight,
        "tax": tax,
        "total": total
    }

    # Update DB
    db["purchase_orders"].update_one(
        {"po_number": po_number},
        {
            "$set": {
                "items": updated_items,
                "summary": updated_summary
            }
        }
    )

    return {
        "success": True,
        "message": "PO items and summary updated",
        "updated_items": updated_items,
        "summary": updated_summary
    }

@app.post("/po/resolve-red-flag")
def resolve_red_flag(req: dict = Body(...)):
    po_number = req.get("po_number")
    resolved_flag = req.get("resolved_flag")
    if not po_number or not resolved_flag:
        raise HTTPException(status_code=400, detail="Missing po_number or resolved_flag")

    db["purchase_orders"].update_one(
        {"po_number": po_number},
        {"$pull": {"red_flags": resolved_flag}}
    )
    db["notifications"].update_many(
        {"po_number": po_number, "message": resolved_flag},
        {"$set": {"status": "resolved", "resolved_at": datetime.utcnow()}}
    )
    return {"status": "Red flag resolved"}



@app.post("/ai/compare-po-invoice")
async def compare_po_invoice(
    request: Request,
    po_number: str = Form(None),
    invoice_file: UploadFile = File(None)
):
    # Handle JSON payload fallback
    if po_number is None:
        try:
            data = await request.json()
            po_number = data.get("po_number")
        except:
            raise HTTPException(status_code=400, detail="po_number not provided")

    # Fetch PO from DB
    po = db["purchase_orders"].find_one({"po_number": po_number})
    if not po:
        raise HTTPException(status_code=404, detail="PO not found")

    invoice_items = []
    invoice_source = "Database"

    # If PDF invoice file uploaded, parse via OCR + AI
    if invoice_file:
        invoice_source = "Uploaded PDF"
        pdf_content = await invoice_file.read()
        pdf_text = ""

        import fitz  # PyMuPDF

        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            for page in doc:
                pdf_text += page.get_text()

        parse_prompt = f"""
        You are an invoice parser AI. Convert the following extracted invoice text into structured JSON with this format:

        {{
          "items": [
            {{
              "description": "...",
              "qty": <int>,
              "unit_price": <float>
            }}
          ]
        }}

        Invoice text:
        {pdf_text}
        """

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": parse_prompt}],
            temperature=0.0
        )
        parsed = json.loads(response.choices[0].message.content)
        invoice_items = parsed.get("items", [])

    else:
        # Fallback: fetch invoice from DB
        invoice_record = db["invoices"].find_one({"po_number": po_number})
        if not invoice_record:
            raise HTTPException(status_code=404, detail="Invoice not found for this PO")
        invoice = invoice_record.get("invoice", invoice_record)
        invoice_items = invoice.get("items", [])

    po_items = po.get("items", [])
    mismatches = []

    # Compare PO items vs Invoice items
    for po_item in po_items:
        desc = po_item.get("description")
        po_qty = po_item.get("qty")
        po_unit_price = po_item.get("unit_price")

        matched_invoice_item = next(
            (item for item in invoice_items if desc.lower() in item.get("description", "").lower()), None
        )

        if matched_invoice_item:
            inv_qty = matched_invoice_item.get("qty")
            inv_unit_price = matched_invoice_item.get("unit_price")
            if po_qty != inv_qty:
                mismatches.append(f"Quantity mismatch for '{desc}': PO has {po_qty}, Invoice has {inv_qty}")
            if round(po_unit_price, 2) != round(inv_unit_price, 2):
                mismatches.append(f"Unit price mismatch for '{desc}': PO has {po_unit_price}, Invoice has {inv_unit_price}")
        else:
            mismatches.append(f"Item '{desc}' found in PO but missing in invoice")

    # Check for extra items in invoice
    for inv_item in invoice_items:
        desc = inv_item.get("description")
        matched_po_item = next((item for item in po_items if desc.lower() in item.get("description", "").lower()), None)
        if not matched_po_item:
            mismatches.append(f"Item '{desc}' found in Invoice but missing in PO")

    # Generate red flag notification and save to purchase_orders if mismatches found
    if mismatches:
        red_flag_entry = {
            "po_number": po_number,
            "type": "Red Flag",
            "message": "PO–Invoice Mismatch Detected",
            "details": mismatches,
            "ai_suggestion": "Review and resolve discrepancies before payment processing.",
            "timestamp": datetime.utcnow(),
            "status": "unread"
        }

        # Insert into notifications collection
        db["notifications"].insert_one(red_flag_entry)

        # Append to purchase_orders.red_flags array
        db["purchase_orders"].update_one(
            {"po_number": po_number},
            {"$push": {"red_flags": {"$each": mismatches}}}
        )

    return {
        "po_number": po_number,
        "invoice_source": invoice_source,
        "mismatches": mismatches,
        "status": "Mismatch Found" if mismatches else "PO and Invoice match perfectly"
    }

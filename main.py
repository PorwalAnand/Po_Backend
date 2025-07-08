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
import uuid

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

# Helper function to clean single dict
def clean_id(document):
    """Remove MongoDB ObjectId from a single document for API responses"""
    return {k: v for k, v in document.items() if k != '_id'}

# Your endpoint
@app.get("/po/{po_number}")
def get_po_detail(po_number: str):
    po = db["purchase_orders"].find_one({"po_number": po_number})
    emails = list(db["emails"].find({"po_number": po_number}))
    if not po:
        raise HTTPException(status_code=404, detail="PO not found")
    check_and_create_red_flag_notification(po)
    return {
        "po": clean_id(po),  # use clean_id here
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


def clean_id(document):
    """Remove MongoDB ObjectId from a single document for API responses"""
    return {k: v for k, v in document.items() if k != '_id'}


@app.post("/ai/invoice")
def generate_invoice(req: InvoiceRequest):
    po = db["purchase_orders"].find_one({"po_number": req.po_number})
    emails = list(db["emails"].find({"po_number": req.po_number}))
    
    if not po:
        raise HTTPException(status_code=404, detail="PO not found")

    po_clean = clean_id(po)

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
    try:
        # Basic validation
        if not req.po_number:
            raise HTTPException(status_code=400, detail="PO number is required")
        
        po = db["purchase_orders"].find_one({"po_number": req.po_number})
        emails = list(db["emails"].find({"po_number": req.po_number}))

        if not po:
            raise HTTPException(status_code=404, detail="PO not found")
        if not emails:
            raise HTTPException(status_code=404, detail="No emails found for this PO")

        # Sort emails chronologically - CRITICAL for proper analysis
        emails.sort(key=lambda e: e.get("date", ""), reverse=False)

        # Get essential PO data
        po_items = po.get("items", [])
        current_po_status = po.get("status", "Unknown")
        current_payment_status = po.get("payment_status", "Unknown")
        
        # Get existing red flags
        existing_red_flags = po.get("red_flags", [])
        last_analysis_date = po.get("last_ai_analysis")
        
        # Get last email date to check if new emails exist
        last_email_date = max([email.get("date", "") for email in emails]) if emails else None
        
        # Check if we have active existing red flags
        active_existing_flags = [flag for flag in existing_red_flags if flag.get("status") == "active"]
        resolved_flag_issues = [flag.get("issue") for flag in existing_red_flags if flag.get("status") == "resolved"]
        
        # Create item context
        item_context = "\n".join([
            f"- {item['description']} (qty: {item['qty']}, unit_price: {item['unit_price']})" 
            for item in po_items
        ])

        # Get current timestamp as string for consistent serialization
        current_timestamp = datetime.utcnow().isoformat()

        # Check if we should skip AI analysis (no new emails since last analysis)
        if (last_analysis_date and last_email_date and 
            last_email_date <= last_analysis_date.isoformat() and 
            active_existing_flags):
            
            # Format existing flags for response
            formatted_existing_flags = []
            for flag in active_existing_flags:
                formatted_flag = {
                    "category": flag.get("category", "Other"),
                    "issue": flag.get("issue", ""),
                    "evidence": flag.get("evidence", ""),
                    "ai_suggestion": flag.get("ai_suggestion", ""),
                    "stakeholder": flag.get("stakeholder", "Both"),
                    "blocks_completion": flag.get("blocks_completion", "")
                }
                formatted_existing_flags.append(formatted_flag)

            # Return existing red flags without new AI analysis
            return {
                "po_number": req.po_number,
                "status": "No new emails - returning existing red flags",
                "should_raise_flags": len(active_existing_flags) > 0,
                "red_flags_count": len(active_existing_flags),
                "new_flags_count": 0,
                "existing_flags_count": len(active_existing_flags),
                "red_flags": formatted_existing_flags,
                "new_red_flags": [],
                "po_status": {
                    "current_status": current_po_status,
                    "key_issues": "No new updates",
                    "next_actions": "Continue monitoring",
                    "last_analysis": current_timestamp,
                    "red_flags_count": len(active_existing_flags),
                    "new_flags_count": 0,
                    "existing_flags_count": len(active_existing_flags)
                },
                "red_flags_details": active_existing_flags,
                "analysis_timestamp": current_timestamp,
                "analysis_type": "existing_flags_returned"
            }

        # Prepare existing issues context for AI
        existing_issues_context = ""
        if active_existing_flags:
            existing_issues_context = f"""
**EXISTING ACTIVE RED FLAGS (Do not duplicate these):**
{chr(10).join([f"- {flag['issue']}" for flag in active_existing_flags])}
"""

        resolved_issues_context = ""
        if resolved_flag_issues:
            resolved_issues_context = f"""
**RESOLVED ISSUES (Do not flag these again):**
{chr(10).join([f"- {issue}" for issue in resolved_flag_issues])}
"""

        # YOUR PROVEN PROMPT - INTEGRATED CAREFULLY
        prompt = f"""
You are a Procurement Manager AI Agent operating within a manufacturing company's purchase order management system. Your primary objective is to analyze purchase order (PO) communication threads and identify any obstacles that prevent successful PO completion, defined as full item delivery and complete payment processing.

Core Responsibilities
1. PO Status Assessment: Evaluate current purchase order status based on provided information
2. Email Thread Analysis: Parse each email sequentially to identify discrepancies, blockers, or potential issues
3. Issue Classification: Categorize identified problems using standardized taxonomy
4. Risk Flagging: Generate red flags for explicit hindrances that block PO progression
5. Solution Recommendation: Provide actionable solutions for identified issues

Issue Classification Taxonomy

Primary Categories
- PO_Details_Issue: Missing/unclear tax numbers, client information, addresses, contact details, legal compliance gaps
- PO_Notes_Issue: Unactioned customizations, special requests, quantity mismatches, specification changes
- Payment_Issue: Payment terms discrepancies, transaction delays, missing confirmations, partial/delayed/stopped payments
- Delivery_Issue: Delivery delays, early delivery conflicts, cargo damage, delivery-invoice discrepancies
- Item_Detail_Issue: Description mismatches, quantity errors, unit price discrepancies, style/color/branding/size issues
- Manufacturing_Problem: Production line issues, material shortages, quality rejections, capacity constraints
- Conflict: Client-manufacturer instruction misalignments, contradictory requirements
- Other: Any explicit issue not covered by above categories that blocks delivery or payment

Issue Validation Rules
- Explicit Content Only: Flag only issues explicitly stated in email content
- No Inference: Do not assume or infer unstated problems
- Mutual Agreement Exception: Do not flag items where both client and manufacturer have explicitly approved changes and confirmed processing
- Completion Blocking: Only flag issues that demonstrably prevent delivery or payment completion

Analysis Methodology

Email Processing Chain
For each email in the thread, execute this sequential analysis:

1. Content Extraction: What is the primary message and intent?
2. Issue Identification: Does content indicate delivery/payment blocking issues?
3. Category Classification: Which taxonomy category applies?
4. Evidence Gathering: What exact quote demonstrates the problem?
5. Stakeholder Assignment: Who is responsible (Client/Manufacturer/Both)?
6. Solution Formulation: What specific actionable solution resolves the issue?
7. Impact Assessment: How does this block PO completion?

Critical Analysis Points
- Parse emails chronologically in provided order
- Identify explicit statements of problems, delays, or conflicts
- Distinguish between resolved issues and active blockers
- Recognize mutual agreements and confirmed changes
- Focus on completion-critical obstacles only

Output Specification
Required JSON Structure
{{
  "should_raise_flags": boolean,
  "red_flags": [
    {{
      "category": "enum_value",
      "issue": "specific_problem_description",
      "evidence": "direct_quote_from_email",
      "ai_suggestion": "actionable_solution",
      "stakeholder": "enum_value",
      "blocks_completion": "completion_impact_explanation"
    }}
  ],
  "po_status": {{
    "current_status": "status_summary",
    "key_issues": "main_hindrances_summary",
    "next_actions": "required_progression_steps"
  }}
}}

Output Requirements
- Format: Valid JSON only, no markdown formatting or explanations
- Content: Professional, clear, actionable language
- Structure: Strict adherence to specified schema
- Completeness: All required fields populated with relevant data
- Accuracy: Evidence-based findings with direct email quotes

Response Validation
- If no hindrances exist: "should_raise_flags": false with empty red_flags array
- All red flags must include completion-blocking justification
- Evidence quotes must be verbatim from source emails
- AI suggestions must be specific and actionable
- Stakeholder assignments must be accurate based on email content

Operational Guidelines

Quality Assurance
- Maintain professional tone in all outputs
- Ensure suggestions are practical and implementable
- Provide clear cause-and-effect relationships
- Avoid speculation or assumption-based conclusions

Production Environment Considerations
- Output must be immediately consumable by downstream systems
- JSON structure must be parser-friendly and error-free
- Response times should be optimized for real-time processing
- Error handling should be graceful and informative

Input Processing Instructions

When provided with PO information and email thread:
1. Review current PO status and context
2. Process emails sequentially in chronological order
3. Apply analysis methodology to each email
4. Aggregate findings into structured output
5. Validate JSON format before response
6. Ensure all critical elements are addressed

{existing_issues_context}

{resolved_issues_context}

**CURRENT PO STATUS:**
- PO Number: {req.po_number}
- Status: {current_po_status}
- Payment Status: {current_payment_status}
- Items: {item_context}

**EMAIL THREAD (Process chronologically):**
{json.dumps(clean_ids(emails), indent=2)}

CRITICAL REMINDERS:
- Only flag issues explicitly stated in email content
- Do not infer or assume unstated problems
- Evidence quotes must be verbatim from emails
- AI suggestions must be specific and actionable
- Stakeholder assignments must be accurate (Client/Manufacturer/Both)
- Focus only on completion-blocking obstacles
- Return valid JSON only, no markdown or explanations

Remember: Your role is to be a precise, analytical, and solution-oriented procurement professional who identifies real obstacles to PO completion and provides actionable paths forward.
"""

        # AI Analysis with optimized settings for your proven prompt
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Slightly higher than original for better detection
                max_tokens=3000,  # Increased for comprehensive analysis
                top_p=0.95,  # High precision for factual analysis
                frequency_penalty=0.1,  # Slight penalty to avoid repetition
                presence_penalty=0.1  # Encourage comprehensive coverage
            )

            ai_content = response.choices[0].message.content.strip()
            
            # Enhanced JSON cleaning and validation - CRITICAL for your prompt
            if ai_content.startswith("json"):
                ai_content = ai_content.replace("json", "").replace("", "").strip()
            elif ai_content.startswith(""):
                ai_content = ai_content.replace("", "").strip()
            
            # Remove any markdown formatting that might interfere
            ai_content = ai_content.replace("**", "").replace("*", "")
            
            # Robust JSON parsing with multiple fallback strategies
            try:
                ai_result = json.loads(ai_content)
            except json.JSONDecodeError:
                # Enhanced JSON extraction for your prompt format
                import re
                
                # Try to find complete JSON object
                json_pattern = r'\{(?:[^{}]*(?:\{[^{}]*\}[^{}]*)*)*\}'
                json_matches = re.findall(json_pattern, ai_content, re.DOTALL)
                
                if json_matches:
                    # Try each JSON match until one parses correctly
                    for json_match in sorted(json_matches, key=len, reverse=True):
                        try:
                            ai_result = json.loads(json_match)
                            break
                        except json.JSONDecodeError:
                            continue
                    else:
                        # If no JSON parses, raise error
                        raise json.JSONDecodeError("No valid JSON found in AI response", ai_content, 0)
                else:
                    raise json.JSONDecodeError("No JSON structure found in AI response", ai_content, 0)
            
            # Validate required top-level fields as per your prompt specification
            required_fields = ["should_raise_flags", "red_flags", "po_status"]
            for field in required_fields:
                if field not in ai_result:
                    if field == "should_raise_flags":
                        ai_result[field] = False
                    elif field == "red_flags":
                        ai_result[field] = []
                    elif field == "po_status":
                        ai_result[field] = {
                            "current_status": current_po_status,
                            "key_issues": "Analysis incomplete - missing po_status",
                            "next_actions": "Review and retry analysis"
                        }
            
            # Validate po_status structure
            po_status = ai_result.get("po_status", {})
            required_po_fields = ["current_status", "key_issues", "next_actions"]
            for field in required_po_fields:
                if field not in po_status:
                    if field == "current_status":
                        po_status[field] = current_po_status
                    elif field == "key_issues":
                        po_status[field] = "No issues identified"
                    elif field == "next_actions":
                        po_status[field] = "Continue monitoring"
            
        except openai.error.OpenAIError as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
        except json.JSONDecodeError as e:
            # Enhanced error logging for debugging
            print(f"JSON Parse Error: {str(e)}")
            print(f"AI Response: {ai_content}")
            raise HTTPException(status_code=500, detail=f"Invalid AI response format: {str(e)}")
        except Exception as e:
            print(f"AI Analysis Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"AI analysis error: {str(e)}")

        # Process results according to your prompt specifications
        should_raise_flags = ai_result.get("should_raise_flags", False)
        
        # Skip processing if PO is complete and paid
        if current_po_status == "Completed" and current_payment_status == "Paid":
            should_raise_flags = False

        processed_red_flags = []
        new_notifications = []

        # Enhanced validation for AI response compliance with your prompt
        if should_raise_flags:
            red_flags = ai_result.get("red_flags", [])
            
            existing_issues = [flag["issue"] for flag in active_existing_flags]
            
            # Valid categories as per your taxonomy
            valid_categories = [
                "PO_Details_Issue", "PO_Notes_Issue", "Payment_Issue", 
                "Delivery_Issue", "Item_Detail_Issue", "Manufacturing_Problem", 
                "Conflict", "Other"
            ]
            
            # Valid stakeholders as per your specification
            valid_stakeholders = ["Client", "Manufacturer", "Both"]
            
            for flag in red_flags:
                # Strict validation following your prompt requirements
                if not isinstance(flag, dict):
                    continue
                
                # Check required fields are present and non-empty (per your output specification)
                required_fields = ["category", "issue", "evidence", "ai_suggestion", "stakeholder", "blocks_completion"]
                if not all(flag.get(field) and str(flag.get(field)).strip() for field in required_fields):
                    continue
                
                flag_issue = flag.get("issue", "").strip()
                flag_evidence = flag.get("evidence", "").strip()
                
                # Skip if already exists or resolved
                if flag_issue in existing_issues or flag_issue in resolved_flag_issues:
                    continue
                
                # Validate category enum
                category = flag.get("category", "Other")
                if category not in valid_categories:
                    category = "Other"
                
                # Validate stakeholder enum
                stakeholder = flag.get("stakeholder", "Both")
                if stakeholder not in valid_stakeholders:
                    stakeholder = "Both"
                
                # Store with enhanced validation per your specification
                flag_data = {
                    "flag_id": str(uuid.uuid4()),
                    "category": category,
                    "priority": "Medium",
                    "issue": flag_issue,
                    "evidence": flag_evidence,
                    "ai_suggestion": flag.get("ai_suggestion", "").strip(),
                    "stakeholder": stakeholder,
                    "blocks_completion": flag.get("blocks_completion", "").strip(),
                    "created_at": datetime.utcnow().isoformat(),
                    "status": "active",
                    "po_status_at_creation": current_po_status,
                    "payment_status_at_creation": current_payment_status
                }
                
                processed_red_flags.append(flag_data)
                
                new_notifications.append({
                    "po_number": req.po_number,
                    "type": "Red Flag",
                    "message": flag_issue,
                    "ai_suggestion": flag_data["ai_suggestion"],
                    "priority": "Medium",
                    "stakeholder": stakeholder,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "active",
                    "flag_id": flag_data["flag_id"]
                })

        # Combine existing active flags with new flags
        all_active_flags = active_existing_flags + processed_red_flags

        # Process PO status with enhanced validation per your specification
        po_status_data = ai_result.get("po_status", {})
        
        status_update = {
            "current_status": po_status_data.get("current_status", current_po_status),
            "key_issues": po_status_data.get("key_issues", "No major issues identified" if not all_active_flags else "Issues identified requiring attention"),
            "next_actions": po_status_data.get("next_actions", "Continue monitoring" if not all_active_flags else "Address identified issues"),
            "last_analysis": current_timestamp,
            "red_flags_count": len(all_active_flags),
            "new_flags_count": len(processed_red_flags),
            "existing_flags_count": len(active_existing_flags)
        }

        # Update database with proper datetime serialization
        update_data = {
            "red_flags": all_active_flags,
            "po_status_analysis": status_update,
            "last_ai_analysis": datetime.utcnow(),
            "should_raise_flags": len(all_active_flags) > 0
        }

        db["purchase_orders"].update_one(
            {"po_number": req.po_number},
            {"$set": update_data}
        )

        # Insert notifications
        if new_notifications:
            db["notifications"].insert_many(new_notifications)

        # Prepare red flags in the desired format (AI structure only)
        formatted_red_flags = []
        for flag in all_active_flags:
            formatted_flag = {
                "category": flag.get("category", "Other"),
                "issue": flag.get("issue", ""),
                "evidence": flag.get("evidence", ""),
                "ai_suggestion": flag.get("ai_suggestion", ""),
                "stakeholder": flag.get("stakeholder", "Both"),
                "blocks_completion": flag.get("blocks_completion", "")
            }
            formatted_red_flags.append(formatted_flag)

        # Create response with consistent structure
        response_data = {
            "po_number": req.po_number,
            "status": "Analysis completed",
            "should_raise_flags": len(all_active_flags) > 0,
            "red_flags_count": len(all_active_flags),
            "new_flags_count": len(processed_red_flags),
            "existing_flags_count": len(active_existing_flags),
            "red_flags": formatted_red_flags,
            "new_red_flags": [flag["issue"] for flag in processed_red_flags],
            "po_status": status_update,
            "red_flags_details": all_active_flags,
            "analysis_timestamp": current_timestamp,
            "analysis_type": "new_analysis_with_existing_flags"
        }

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        db["error_logs"].insert_one({
            "po_number": req.po_number if 'req' in locals() else "Unknown",
            "error": str(e),
            "timestamp": datetime.utcnow(),
            "function": "update_po_from_emails"
        })
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/ai/resolve-red-flag")
def resolve_red_flag(req: dict):
    """
    Mark a red flag as resolved and move it to historical_flags.
    Request body: {"po_number": "PO123", "flag_id": "uuid-string"}
    """
    try:
        po_number = req.get("po_number")
        flag_id = req.get("flag_id")
        
        if not po_number or not flag_id:
            raise HTTPException(status_code=400, detail="po_number and flag_id are required")
        
        # Find the PO
        po = db["purchase_orders"].find_one({"po_number": po_number})
        if not po:
            raise HTTPException(status_code=404, detail="PO not found")
        
        red_flags = po.get("red_flags", [])
        historical_flags = po.get("historical_flags", [])
        resolved_flag = None

        # Find and remove the specific red flag from active red_flags
        updated_red_flags = []
        for flag in red_flags:
            if flag.get("flag_id") == flag_id:
                flag["status"] = "resolved"
                flag["resolved_at"] = datetime.utcnow()
                resolved_flag = flag
            else:
                updated_red_flags.append(flag)
        
        if not resolved_flag:
            raise HTTPException(status_code=404, detail="Red flag not found")
        
        # Append resolved flag to historical_flags
        historical_flags.append(resolved_flag)
        
        # Update the PO in DB
        db["purchase_orders"].update_one(
            {"po_number": po_number},
            {
                "$set": {
                    "red_flags": updated_red_flags,
                    "historical_flags": historical_flags,
                    "last_flag_resolved": datetime.utcnow()
                }
            }
        )
        
        # Update notification status
        db["notifications"].update_one(
            {"po_number": po_number, "flag_id": flag_id},
            {"$set": {"status": "resolved", "resolved_at": datetime.utcnow()}}
        )
        
        return {
            "po_number": po_number,
            "flag_id": flag_id,
            "status": "Red flag marked as resolved and moved to historical_flags",
            "active_flags_remaining": len(updated_red_flags),
            "resolved_at": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resolving red flag: {str(e)}")


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

# @app.post("/po/resolve-red-flag")
# def resolve_red_flag(req: dict = Body(...)):
#     po_number = req.get("po_number")
#     resolved_flag = req.get("resolved_flag")
#     if not po_number or not resolved_flag:
#         raise HTTPException(status_code=400, detail="Missing po_number or resolved_flag")

#     db["purchase_orders"].update_one(
#         {"po_number": po_number},
#         {"$pull": {"red_flags": resolved_flag}}
#     )
#     db["notifications"].update_many(
#         {"po_number": po_number, "message": resolved_flag},
#         {"$set": {"status": "resolved", "resolved_at": datetime.utcnow()}}
#     )
#     return {"status": "Red flag resolved"}



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

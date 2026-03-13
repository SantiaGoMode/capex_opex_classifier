#!/usr/bin/env python3
"""
Finance Analyst Agent: CapEx/OpEx Classifier (with Reasoning Report)
Uses Qwen 2.5 32B via Ollama to classify Jira work items against company standards.
"""

import ollama
import pandas as pd
import json
import re
import sys
import os
from datetime import datetime
from typing import List, Dict, Optional

# ================= CONFIGURATION =================
CONFIG = {
    "model": "qwen2.5:32b",
    "standards_file": "work-categories.csv",
    "input_file": "jira-items.csv",
    "output_dir": "output",
    "output_csv": "classified_items.csv",
    "output_txt": "classification_reasoning.txt",
    "temperature": 0.2,
    "num_ctx": 8192,
    "batch_size": 5,
}

# ================= PROMPT TEMPLATES =================
SYSTEM_PROMPT = """
You are a senior Finance Analyst specializing in CapEx/OpEx classification for technology projects.
Your job is to analyze Jira work items against the provided Company Standards Document.

CLASSIFICATION RULES:
1. CapEx (Capital Expenditure): Assets with multi-year benefit, new development, IP creation, infrastructure with >1yr life.
2. OpEx (Operational Expenditure): Maintenance, bug fixes, recurring costs, training, day-to-day operations.
3. KEY SIGNALS: 
   - Match the 'Logic/Rule' keywords (e.g., "Create", "Fix", "Monitor") against the Jira Item Summary.
   - Check if the Jira 'Issue Type' (e.g., Story, Bug) aligns with the allowed 'Jira Issue Types' in the standards.
4. If uncertain, classify as OpEx (conservative accounting principle).
5. Output ONLY valid JSON. No markdown, no explanations outside the JSON.
"""

PHASE_1_PROMPT = """
<standards_document>
{standards_text}
</standards_document>

TASK: Internalize these CapEx/OpEx classification standards. 
Confirm you understand by responding with: {"status": "standards_loaded", "categories_count": N}
"""

PHASE_2_PROMPT = """
<company_standards>
{standards_text}
</company_standards>

<jira_items_batch>
{items_csv}
</jira_items_batch>

TASK: Classify each Jira item as "CapEx" or "OpEx" based strictly on the Company Standards above.

MATCHING LOGIC:
1. Look for keywords from the 'Logic/Rule' column in the Item Summary/Description.
2. Check if the Item 'Issue Type' matches the allowed 'Jira Issue Types' in the standard.
3. If the Item Type is 'Bug', it is almost always OpEx unless it involves a Major Upgrade.

CONFIDENCE SCORING GUIDE:
- 0.90-1.00: Summary keywords AND Issue Type both clearly match a single standard category.
- 0.70-0.89: Keywords match a category but Issue Type is ambiguous, or multiple categories partially match.
- 0.50-0.69: Weak keyword overlap, no clear category match, or conflicting signals (e.g., Bug with "new feature" language).
- Below 0.50: No meaningful match found; defaulting to OpEx.

OUTPUT FORMAT (STRICT JSON):
{{
  "classifications": [
    {{
      "issue_key": "PROJ-101",
      "classification": "CapEx",
      "confidence": 0.92,
      "reasoning": "Summary contains 'Build new payment gateway' matching New Feature Development; Story type is allowed.",
      "matched_category": "New Feature Development",
      "matched_rule": "Create/Build/Enhance"
    }},
    {{
      "issue_key": "PROJ-102",
      "classification": "OpEx",
      "confidence": 0.60,
      "reasoning": "Summary mentions 'update' but context is a minor config change, not a major enhancement. Bug type suggests maintenance.",
      "matched_category": "Maintenance & Support",
      "matched_rule": "Fix/Patch/Monitor"
    }}
  ]
}}

RULES:
- Output ONLY the JSON object.
- "matched_category" must exactly match a Category from the standards.
- "matched_rule" must be one of the rules from that category.
- Vary the confidence score based on the scoring guide above. Do NOT default to the same score for every item.
"""

PHASE_3_PROMPT = """
<company_standards>
{standards_text}
</company_standards>

<classification_results>
{classification_json}
</classification_results>

<jira_item_details>
{item_details}
</jira_item_details>

TASK: Generate a human-readable explanation for each classification decision.
This report will be reviewed by finance auditors, so clarity is critical.

OUTPUT FORMAT (PLAIN TEXT, NO JSON):
================================================================================
ISSUE: {{issue_key}}
SUMMARY: {{summary}}
CLASSIFICATION: {{CapEx/OpEx}}
CONFIDENCE: {{XX}}%
MATCHED CATEGORY: {{category}}
MATCHED RULE: {{rule}}

EXPLANATION:
{{2-3 sentence explanation in plain English suitable for non-technical auditors}}

AUDIT NOTES:
- Key evidence from summary/description
- Why this does/doesn't meet capitalization criteria
- Any flags for manual review
================================================================================

Generate this format for EACH item. Separate items with a blank line.
"""

# ================= HELPER FUNCTIONS =================

def load_standards_formatted(filepath: str) -> str:
    """Loads the standards CSV and formats it into a clean text block for the LLM."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Required file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    required_cols = ['Category', 'Classification', 'Jira Issue Types', 'Description', 'Logic/Rule']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Standards CSV missing columns: {missing}")
    
    formatted_lines = []
    for _, row in df.iterrows():
        issue_types = str(row['Jira Issue Types']).replace('"', '').strip()
        block = (
            f"- Category: {row['Category']}\n"
            f"  Classification: {row['Classification']}\n"
            f"  Allowed Issue Types: {issue_types}\n"
            f"  Description: {row['Description']}\n"
            f"  Logic/Rule Keywords: {row['Logic/Rule']}"
        )
        formatted_lines.append(block)
    
    return "\n\n".join(formatted_lines)

def load_jira_items(filepath: str) -> pd.DataFrame:
    """Loads Jira export CSV with flexible column name handling."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    required = ['Issue Key', 'Summary']
    missing = [c for c in required if c not in df.columns]
    if missing:
        mapping = {'Issue Key': ['Key', 'issue_key', 'ID'], 'Summary': ['Summary', 'summary', 'Title']}
        for std, variations in mapping.items():
            if std not in df.columns:
                for var in variations:
                    if var in df.columns:
                        df.rename(columns={var: std}, inplace=True)
                        break
    
    final_missing = [c for c in required if c not in df.columns]
    if final_missing:
        raise ValueError(f"Jira CSV missing required columns (or aliases): {final_missing}")
        
    return df

def extract_json_from_response(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response, handling markdown wrappers."""
    text = re.sub(r"```(?:json)?\n?", "", text)
    text = re.sub(r"\n```", "", text)
    text = text.strip()
    json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error: {e}")
            return None
    return None

def classify_batch(items_batch: pd.DataFrame, standards_text: str) -> List[Dict]:
    """Send a batch of items to Qwen for classification (Phase 2)."""
    items_csv = items_batch.to_csv(index=False)
    prompt = PHASE_2_PROMPT.format(standards_text=standards_text, items_csv=items_csv)
    
    response = ollama.chat(
        model=CONFIG["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": CONFIG["temperature"],
            "num_ctx": CONFIG["num_ctx"],
            "top_p": 0.85,
            "repeat_penalty": 1.1
        }
    )
    
    result = extract_json_from_response(response["message"]["content"])
    if result and "classifications" in result:
        return result["classifications"]
    else:
        print("⚠️ Failed to parse classifications, using fallback (OpEx).")
        fallback = []
        for _, item in items_batch.iterrows():
            fallback.append({
                "issue_key": item.get("Issue Key", "UNKNOWN"),
                "classification": "OpEx",
                "confidence": 0.5,
                "reasoning": "Fallback: Could not parse LLM response",
                "matched_category": "Unknown",
                "matched_rule": "Unknown"
            })
        return fallback

def generate_reasoning_report(classifications: List[Dict], items_df: pd.DataFrame, standards_text: str) -> str:
    """
    PHASE 3: Generate human-readable reasoning for each classification.
    Returns the full text report as a string.
    """
    print("📝 Phase 3: Generating human-readable reasoning report...")
    
    # Prepare classification results as JSON string for the prompt
    classification_json = json.dumps(classifications, indent=2)
    
    # Prepare item details (merge classifications with original data)
    item_details_list = []
    for classification in classifications:
        key = classification.get("issue_key")
        mask = items_df["Issue Key"] == key
        if mask.any():
            item_row = items_df[mask].iloc[0]
            item_details_list.append({
                "issue_key": key,
                "summary": item_row.get("Summary", "N/A"),
                "issue_type": item_row.get("Issue Type", "N/A"),
                "description": item_row.get("Description", item_row.get("summary", "N/A")),
                "classification": classification.get("classification"),
                "confidence": classification.get("confidence"),
                "matched_category": classification.get("matched_category"),
                "matched_rule": classification.get("matched_rule")
            })
    
    # Convert to readable text format
    item_details_text = ""
    for item in item_details_list:
        item_details_text += f"""
Issue Key: {item['issue_key']}
Summary: {item['summary']}
Issue Type: {item['issue_type']}
Description: {item['description'][:200]}...
Classification: {item['classification']}
Confidence: {item['confidence']}
Matched Category: {item['matched_category']}
Matched Rule: {item['matched_rule']}
---
"""
    
    # Call LLM for Phase 3
    prompt = PHASE_3_PROMPT.format(
        standards_text=standards_text,
        classification_json=classification_json,
        item_details=item_details_text
    )
    
    response = ollama.chat(
        model=CONFIG["model"],
        messages=[
            {"role": "system", "content": "You are a finance analyst writing clear audit explanations. Output plain text only, no JSON."},
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": 0.3,  # Slightly higher for natural language
            "num_ctx": CONFIG["num_ctx"],
        }
    )
    
    return response["message"]["content"]

def save_reasoning_report(report_text: str, output_path: str):
    """Save the reasoning report to a TXT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"💾 Reasoning report saved to: {output_path}")

# ================= MAIN EXECUTION =================

def main():
    print("🔍 Finance Analyst Agent: CapEx/OpEx Classifier")
    print(f"🤖 Model: {CONFIG['model']}")
    print("-" * 60)
    
    # Create output directory
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # 1. Load & Format Standards
    print(f"📚 Loading standards from: {CONFIG['standards_file']}")
    try:
        standards_text = load_standards_formatted(CONFIG["standards_file"])
        print("✅ Standards formatted successfully.")
    except Exception as e:
        print(f"❌ Error loading standards: {e}")
        sys.exit(1)
    
    # 2. Phase 1: Load Context
    print("🧠 Phase 1: Initializing model context...")
    try:
        phase1_response = ollama.chat(
            model=CONFIG["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PHASE_1_PROMPT.format(standards_text=standards_text)}
            ],
            options={"temperature": 0.1, "num_ctx": CONFIG["num_ctx"]}
        )
    except Exception as e:
        print(f"⚠️ Phase 1 warning (continuing anyway): {e}")
    
    # 3. Load Jira Items
    print(f"📋 Loading work items from: {CONFIG['input_file']}")
    try:
        items_df = load_jira_items(CONFIG["input_file"])
        print(f"📊 Found {len(items_df)} items to classify")
    except Exception as e:
        print(f"❌ Error loading Jira items: {e}")
        sys.exit(1)
    
    # 4. Phase 2: Classify in Batches
    print("\n🔧 Phase 2: Classifying work items...")
    all_classifications = []
    batch_size = CONFIG["batch_size"]
    
    for i in range(0, len(items_df), batch_size):
        batch = items_df.iloc[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(items_df)+batch_size-1)//batch_size}")
        
        classifications = classify_batch(batch, standards_text)
        all_classifications.extend(classifications)
    
    # 5. Merge Results into DataFrame
    print("\n🔗 Merging classifications with original data...")
    results_df = items_df.copy()
    results_df["classification"] = "Unknown"
    results_df["confidence"] = 0.0
    results_df["reasoning"] = ""
    results_df["matched_category"] = ""
    results_df["matched_rule"] = ""
    
    for classification in all_classifications:
        key = classification.get("issue_key")
        mask = results_df["Issue Key"] == key
        if mask.any():
            results_df.loc[mask, "classification"] = classification.get("classification", "Unknown")
            results_df.loc[mask, "confidence"] = classification.get("confidence", 0.0)
            results_df.loc[mask, "reasoning"] = classification.get("reasoning", "")
            results_df.loc[mask, "matched_category"] = classification.get("matched_category", "")
            results_df.loc[mask, "matched_rule"] = classification.get("matched_rule", "")
    
    # 6. Save CSV Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_output_path = os.path.join(CONFIG["output_dir"], f"{timestamp}_{CONFIG['output_csv']}")
    results_df.to_csv(csv_output_path, index=False)
    print(f"💾 CSV results saved to: {csv_output_path}")
    
    # 7. PHASE 3: Generate Reasoning Report
    print("\n" + "=" * 60)
    print("📄 PHASE 3: Generating Human-Readable Reasoning Report")
    print("=" * 60)
    
    try:
        reasoning_report = generate_reasoning_report(all_classifications, items_df, standards_text)
        
        txt_output_path = os.path.join(CONFIG["output_dir"], f"{timestamp}_{CONFIG['output_txt']}")
        save_reasoning_report(reasoning_report, txt_output_path)
        
    except Exception as e:
        print(f"⚠️ Phase 3 failed: {e}")
        print("CSV results are still saved. Reasoning report skipped.")
        txt_output_path = None
    
    # 8. Summary
    capex_count = results_df["classification"].eq("CapEx").sum()
    opex_count = results_df["classification"].eq("OpEx").sum()
    low_confidence = results_df[results_df["confidence"] < 0.7]
    
    print("\n" + "=" * 60)
    print("📊 FINAL CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"✅ Total items processed: {len(results_df)}")
    print(f"💰 CapEx eligible: {capex_count} ({capex_count/len(results_df)*100:.1f}%)")
    print(f"🔧 OpEx eligible: {opex_count} ({opex_count/len(results_df)*100:.1f}%)")
    
    if len(low_confidence) > 0:
        print(f"⚠️ Low confidence (<70%): {len(low_confidence)} items - Review recommended")
        print("   Items:", low_confidence["Issue Key"].tolist())
    
    print(f"💾 CSV Output: {csv_output_path}")
    if txt_output_path:
        print(f"📝 Reasoning Report: {txt_output_path}")
    print("=" * 60)
    
    # Show sample
    print("\n📋 Sample Results:")
    cols = ["Issue Key", "Summary", "Issue Type", "classification", "confidence", "matched_category"]
    available_cols = [c for c in cols if c in results_df.columns]
    print(results_df[available_cols].head(3).to_string())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
import os
import json
import glob
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ==========================================
# CONFIGURATION
# ==========================================

# Replace with your actual Gemini API Key or set it in your environment properties
# os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY_HERE"

IMAGE_FOLDER = "catalogue_images"  # Place all your JPG/PNG files in this folder
OUTPUT_FOLDER = "json_output"

# ==========================================
# DEFINE THE EXPECTED JSON STRUCTURE
# ==========================================

class Product(BaseModel):
    category: str = Field(description="The main category of the part, e.g., 'Cylinder Head' or 'Starter Motor'")
    model: str = Field(description="The vehicle model this part belongs to, e.g., 'EECO' or 'Pulsar 200'")
    ref_no: str = Field(description="The figure reference number, e.g., '1-1' or '2'")
    part_no: str = Field(description="The exact alphanumeric part number")
    description: str = Field(description="The name or description of the part")
    qty: str = Field(description="The quantity required for the assembly")
    remarks: str | None = Field(default=None, description="Any extra remarks, like 'GASOLINE' or 'CNG'")

class CatalogPage(BaseModel):
    products: list[Product] = Field(description="A list of all the parts found in the catalog table.")

# ==========================================
# MAIN SCRIPT
# ==========================================

def extract_data_from_folder(folder_path: str, output_folder: str):
    """Uses Gemini 2.5 Flash to extract catalog tables from an entire folder of images into individual JSON files."""
    
    # Check if the API key is set
    if "GEMINI_API_KEY" not in os.environ:
        print("ERROR: Please set the GEMINI_API_KEY environment variable.")
        print("Example (Windows CMD): set GEMINI_API_KEY=your_key_here")
        return

    if not os.path.exists(folder_path):
        print(f"ERROR: Folder not found at '{folder_path}'. Creating an empty folder for you...")
        os.makedirs(folder_path)
        print(f"Please place your catalog images inside '{folder_path}' and run again.")
        return

    # Find all common image types
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
    if not image_files:
        print(f"No images found in '{folder_path}'. Please add JPG or PNG files.")
        return

    print("Initializing Gemini Client...")
    client = genai.Client()
    
    prompt = (
        "You are an expert at transcribing automotive parts catalogs. "
        "Extract every row from the table in the provided image. "
        "Pay attention to the page headers to determine the 'category' and 'model' for all the parts on this page. "
        "For example, if the page says 'CYLINDER HEAD' and 'EECO', use those for category and model."
    )

    print(f"Found {len(image_files)} images to process.\n")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    total_parts_extracted = 0

    for i, image_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] Analyzing {image_path}...")
        
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                
            # Most extensions map closely to their mime_type, defaulting to jpeg
            mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
            
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

            # Call Gemini Flash with Structured Outputs (guarantees the exact JSON format)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CatalogPage,
                    temperature=0.1, # Low temp for data extraction accuracy
                ),
            )

            # Parse the structured JSON response
            page_data = json.loads(response.text)
            products_on_page = page_data.get("products", [])
            total_parts_extracted += len(products_on_page)
            
            # Save the individual JSON file
            base_name = os.path.basename(image_path)
            file_name, _ = os.path.splitext(base_name)
            output_file_path = os.path.join(output_folder, f"{file_name}.json")
            
            with open(output_file_path, "w") as f:
                # Wrap it in {"products": [...]} for consistency
                json.dump({"products": products_on_page}, f, indent=4)
                
            print(f"    -> Extracted {len(products_on_page)} parts. Saved to {output_file_path}")
            
        except Exception as e:
            print(f"    -> ERROR processing {image_path}: {e}")
        
    print(f"\n Processing Complete!")
    print(f"Total parts extracted across all images: {total_parts_extracted}")
    print(f"Individual JSON files saved to: {output_folder}/")


if __name__ == "__main__":
    extract_data_from_folder(IMAGE_FOLDER, OUTPUT_FOLDER)
